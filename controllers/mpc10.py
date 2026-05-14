"""
MPC controller for the 10-state wheel-leg robot.

Drop-in replacement for LQR10Controller sharing the exact same:
  - Observation vector (10 states): [s, ds, phi, dphi, theta_ll, dtheta_ll,
                                     theta_lr, dtheta_lr, theta_b, dtheta_b]
  - Control vector  (4 inputs):     [T_wl, T_wr, T_bl, T_br]
  - Sign conventions (mirrored from lqr10.py)
  - Leg PD law      (K_P=2500, K_D=120, locked at L=0.15 m)
  - Linearization   (get_AB10 + discretize from controllers.model10)

The controller solves a 20-step QP at 100 Hz (DT=0.01 s) on top of a 500 Hz
physics simulation; the LQR equivalent solves the steady-state Riccati once
offline. Both call back into the same model10 linearization at L_l=L_r=0.15
so behaviour can be compared directly.

Output MuJoCo ctrl: [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]
"""

import numpy as np
import scipy.linalg
import cvxpy as cp

from .params import R_W, L_MIN, L_MAX
from .model10 import get_AB10, discretize
from sim.state import extract_state, get_joint_ids
from sim.command import ControlCommand


# ── Sign conventions (now match controllers/lqr10.py exactly) ────────────
# After fixing model10.py to match MATLAB's A, B exactly, scipy's continuous
# ARE solution on (A, B, Q, R) agrees with MATLAB icare's K_user to 5
# decimal places. The same LQR-validated MuJoCo signs therefore apply.
_SIGN_S        = +1
_SIGN_PHI      = -1
_SIGN_THETA_LL = +1
_SIGN_THETA_B  = -1
_SIGN_HIP_OUT  = +1
_SIGN_WHL_OUT  = -1

# ── Leg PD gains (must match controllers/lqr10.py exactly) ───────────────
_K_LEG_P  = 2500.0
_K_LEG_D  = 120.0
_LEG_FF   = 0.0


def _leg_pd(L_ref: float, L_meas: float, L_dot: float) -> float:
    ext_ref  = L_ref  - L_MIN
    ext_meas = L_meas - L_MIN
    return _K_LEG_P * (ext_ref - ext_meas) - _K_LEG_D * L_dot + _LEG_FF


def _compute_leg_refs(h_ref: float, theta_ll: float, theta_lr: float):
    cos_l = max(np.cos(theta_ll), 0.1)
    cos_r = max(np.cos(theta_lr), 0.1)
    L_l = float(np.clip((h_ref - R_W) / cos_l, L_MIN, L_MAX))
    L_r = float(np.clip((h_ref - R_W) / cos_r, L_MIN, L_MAX))
    return L_l, L_r


class MPC10Controller:
    """20-step linear MPC at 100 Hz, locked at L=0.15 m leg length."""

    N     = 20
    DT    = 0.01                         # 100 Hz solve rate
    # Same Q, R as LQR's MATLAB icare run (LQR.m)
    Q     = np.diag([10.0, 300.0, 5000.0, 1.0,
                     5000.0, 1.0, 5000.0, 1.0,
                     25000.0, 1.0])
    R     = np.diag([40.0, 40.0, 1.0, 1.0])
    U_MAX = np.array([30.0, 30.0, 15.0, 15.0])    # [T_wl, T_wr, T_bl, T_br]
    LEG_LEN = 0.15                       # locked operating point

    def __init__(self):
        # ── Linearize once at the locked leg length ──────────────────────
        A_c, B_c = get_AB10(self.LEG_LEN, self.LEG_LEN)
        self.Ad, self.Bd = discretize(A_c, B_c, self.DT)
        # Terminal cost = discrete-time ARE solution → infinite-horizon tail
        self.P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

        # ── Build CVXPY problem once (DPP-compliant for OSQP warm start) ─
        self._build_problem()

        # ── Cached state ────────────────────────────────────────────────
        self._joint_ids   = None
        self._last_u      = np.zeros(4)
        self._last_ctrl   = np.zeros(6)
        self._last_solve_t = -np.inf

    # ── QP build ─────────────────────────────────────────────────────────
    def _build_problem(self):
        n, m, N = 10, 4, self.N
        self._x_var = cp.Variable((n, N + 1))
        self._u_var = cp.Variable((m, N))

        self._x0_p  = cp.Parameter(n,            name="x0")
        self._ref_p = cp.Parameter((n, N + 1),   name="ref")

        Q_chol = np.linalg.cholesky(self.Q)
        R_chol = np.linalg.cholesky(self.R)
        P_chol = np.linalg.cholesky(self.P)

        cost = 0
        cons = [self._x_var[:, 0] == self._x0_p]

        for k in range(N):
            e_k = self._x_var[:, k] - self._ref_p[:, k]
            cost += cp.sum_squares(Q_chol @ e_k)
            cost += cp.sum_squares(R_chol @ self._u_var[:, k])
            cons += [
                self._x_var[:, k+1] == self.Ad @ self._x_var[:, k]
                                       + self.Bd @ self._u_var[:, k],
                self._u_var[0, k] <=  self.U_MAX[0], self._u_var[0, k] >= -self.U_MAX[0],
                self._u_var[1, k] <=  self.U_MAX[1], self._u_var[1, k] >= -self.U_MAX[1],
                self._u_var[2, k] <=  self.U_MAX[2], self._u_var[2, k] >= -self.U_MAX[2],
                self._u_var[3, k] <=  self.U_MAX[3], self._u_var[3, k] >= -self.U_MAX[3],
            ]

        e_N = self._x_var[:, N] - self._ref_p[:, N]
        cost += cp.sum_squares(P_chol @ e_N)

        self._prob = cp.Problem(cp.Minimize(cost), cons)
        self._first_solve = True

    # ── Solve ────────────────────────────────────────────────────────────
    def _solve(self, x0: np.ndarray, ref: np.ndarray) -> np.ndarray:
        self._x0_p.value  = x0
        self._ref_p.value = ref
        try:
            self._prob.solve(
                solver=cp.OSQP,
                warm_start=not self._first_solve,
                verbose=False,
                max_iter=10000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=False,
                adaptive_rho=True,
                rho=0.1,
            )
            if self._u_var.value is not None:
                self._last_u = self._u_var.value[:, 0].copy()
            self._first_solve = False
        except Exception:
            self._first_solve = False
        return self._last_u.copy()

    # ── Main interface ───────────────────────────────────────────────────
    def compute(self, model, data, cmd: ControlCommand) -> np.ndarray:
        """
        Called from sim/runner.py at the runner's DT_CTRL rate (500 Hz today).
        Internally rate-limits the QP solve to self.DT (10 ms / 100 Hz);
        between solves we return the cached ctrl so the wheel/hip torques
        hold over the simulation's intermediate physics steps.
        """
        if self._joint_ids is None:
            self._joint_ids = get_joint_ids(model)

        t = float(data.time)
        # Skip QP solve if we already solved within the last DT seconds
        if t - self._last_solve_t < self.DT - 1e-9:
            # Still refresh leg PD so it tracks fast (it's stateless on dt)
            return self._leg_pd_refresh(model, data, cmd)

        # ── Build x0 in LQR/MATLAB sign convention ────────────────────────
        st = extract_state(model, data, self._joint_ids)
        x0 = np.array([
            st["s"]        * _SIGN_S,        st["ds"]        * _SIGN_S,
            st["phi"]      * _SIGN_PHI,      st["dphi"]      * _SIGN_PHI,
            st["theta_ll"] * _SIGN_THETA_LL, st["dtheta_ll"] * _SIGN_THETA_LL,
            st["theta_lr"] * _SIGN_THETA_LL, st["dtheta_lr"] * _SIGN_THETA_LL,
            st["theta_b"]  * _SIGN_THETA_B,  st["dtheta_b"]  * _SIGN_THETA_B,
        ])

        # Constant reference over horizon
        ref = np.zeros((10, self.N + 1))
        ref[1, :] = cmd.vx_ref
        ref[2, :] = cmd.yaw_ref * _SIGN_PHI

        # ── Solve QP and unpack ───────────────────────────────────────────
        u_mpc = self._solve(x0, ref)
        T_wl, T_wr, T_bl, T_br = u_mpc

        # Leg PD (identical to LQR)
        L_ref_l, L_ref_r = _compute_leg_refs(cmd.h_ref,
                                              st["theta_ll"], st["theta_lr"])
        F_l = _leg_pd(L_ref_l, st["L_l"], st["leg_L_dot"])
        F_r = _leg_pd(L_ref_r, st["L_r"], st["leg_R_dot"])

        self._last_ctrl = np.array([
            T_bl * _SIGN_HIP_OUT,
            T_br * _SIGN_HIP_OUT,
            F_l, F_r,
            T_wl * _SIGN_WHL_OUT,
            T_wr * _SIGN_WHL_OUT,
        ])
        self._last_solve_t = t
        return self._last_ctrl

    def _leg_pd_refresh(self, model, data, cmd):
        """Cheap update: keep MPC torques but refresh leg PD every call."""
        st = extract_state(model, data, self._joint_ids)
        L_ref_l, L_ref_r = _compute_leg_refs(cmd.h_ref,
                                              st["theta_ll"], st["theta_lr"])
        F_l = _leg_pd(L_ref_l, st["L_l"], st["leg_L_dot"])
        F_r = _leg_pd(L_ref_r, st["L_r"], st["leg_R_dot"])
        ctrl = self._last_ctrl.copy()
        ctrl[2] = F_l
        ctrl[3] = F_r
        return ctrl

    @property
    def last_u(self) -> np.ndarray:
        """Last solved [T_wl, T_wr, T_bl, T_br] — for runner logging."""
        return self._last_u.copy()
