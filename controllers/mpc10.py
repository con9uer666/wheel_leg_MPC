"""
MPC controller for the 10-state wheel-leg robot.

State:  x = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]
Input:  u = [T_wl, T_wr, T_bl, T_br]
Output: MuJoCo ctrl = [T_bl, T_br, F_leg_L, F_leg_R, T_wl, T_wr]
"""

import numpy as np
import scipy.linalg
import cvxpy as cp

from .params import G, R_W, M_B, L_MIN, L_MAX
from .model10 import get_AB10, discretize
from sim.state import extract_state, state_to_vec, get_joint_ids
from sim.command import ControlCommand

# ── Cost matrices (from MATLAB lqr_Q / lqr_R) ─────────────────────────────
# theta_ll / theta_lr weights set to 0: their coupling with T_bl and theta_b
# through theta_ll=-(hip+pitch) causes the MPC to drive the hip to joint limits.
# Body pitch (theta_b) and velocity (ds) are the primary balance objectives.
_Q_DIAG = np.array([10.0, 20.0, 12000.0, 200.0,
                     0.01, 0.01, 0.01, 0.01,
                     20000.0, 100.0])
_R_DIAG = np.array([0.25, 0.25, 1.5, 1.5])   # [T_wl, T_wr, T_bl, T_br]

# ── Actuator limits (must match wheel_legged.xml) ─────────────────────────
_U_MAX = np.array([30.0, 30.0, 15.0, 15.0])    # [T_wl, T_wr, T_bl, T_br] – wheel 30 Nm, hip 15 Nm

# ── Leg PD gains ──────────────────────────────────────────────────────────
_K_LEG_P  = 2500.0
_K_LEG_D  = 120.0
_LEG_FF   = 0.0  # prismatic joint damping=5.0 handles gravity; no feedforward needed


def _leg_pd(L_ref: float, L_meas: float, L_dot: float) -> float:
    """PD + gravity feedforward for one prismatic leg actuator."""
    ext_ref  = L_ref  - L_MIN
    ext_meas = L_meas - L_MIN
    return _K_LEG_P * (ext_ref - ext_meas) - _K_LEG_D * L_dot + _LEG_FF


def _compute_leg_refs(h_ref: float, theta_ll: float, theta_lr: float):
    """Desired leg lengths from commanded body height and current leg angles."""
    cos_l = max(np.cos(theta_ll), 0.1)
    cos_r = max(np.cos(theta_lr), 0.1)
    L_l = float(np.clip((h_ref - R_W) / cos_l, L_MIN, L_MAX))
    L_r = float(np.clip((h_ref - R_W) / cos_r, L_MIN, L_MAX))
    return L_l, L_r


class MPC10Controller:
    """
    Full 10-state MPC for the wheel-leg robot.

    Horizon N=20, dt=0.005 s (200 Hz control rate).

    The CVXPY problem is rebuilt when Ad/Bd/P change (on relinearization,
    triggered when leg lengths drift >5mm from the current operating point).
    x0 and ref are CVXPY Parameters so warm-start works between control steps.
    """

    N     = 15
    DT    = 0.02         # control period (s) – 50 Hz
    Q     = np.diag(_Q_DIAG)
    R     = np.diag(_R_DIAG)
    U_MAX = _U_MAX

    def __init__(self):
        self._L_op_l = 0.18
        self._L_op_r = 0.18
        self._last_u = np.zeros(4)
        self._joint_ids = None

        self._relinearize(0.18, 0.18, force=True)   # sets Ad, Bd, P
        self._build_problem()                         # builds QP

    # ── Linearization ────────────────────────────────────────────────────
    def _relinearize(self, L_l: float, L_r: float, force: bool = False):
        changed = (
            force
            or abs(L_l - self._L_op_l) >= 0.005
            or abs(L_r - self._L_op_r) >= 0.005
        )
        if not changed:
            return
        self._L_op_l = L_l
        self._L_op_r = L_r
        A_c, B_c = get_AB10(L_l, L_r)
        self.Ad, self.Bd = discretize(A_c, B_c, self.DT)
        self.P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self._build_problem()   # rebuild with new Ad, Bd, P embedded as constants

    # ── CVXPY problem ──────────────────────────────────────────────────────
    def _build_problem(self):
        """
        Build QP with DPP-compliant structure:
          - Ad, Bd, P, Q, R embedded as numeric constants (not Parameters)
          - x0, ref are CVXPY Parameters (appear only in linear/affine terms)
        This allows OSQP warm-start across control steps (Q, P structure fixed).
        """
        n, m, N = 10, 4, self.N
        self._x_var = cp.Variable((n, N + 1))
        self._u_var = cp.Variable((m, N))

        self._x0_p  = cp.Parameter(n,          name="x0")
        self._ref_p = cp.Parameter((n, N + 1), name="ref")

        # Use Cholesky factors for sum_squares – DPP compliant and efficient
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
                self._x_var[:, k+1] == self.Ad @ self._x_var[:, k] + self.Bd @ self._u_var[:, k],
                self._u_var[0, k] <=  self.U_MAX[0], self._u_var[0, k] >= -self.U_MAX[0],
                self._u_var[1, k] <=  self.U_MAX[1], self._u_var[1, k] >= -self.U_MAX[1],
                self._u_var[2, k] <=  self.U_MAX[2], self._u_var[2, k] >= -self.U_MAX[2],
                self._u_var[3, k] <=  self.U_MAX[3], self._u_var[3, k] >= -self.U_MAX[3],
                self._x_var[8, k] <=  np.deg2rad(40),
                self._x_var[8, k] >= -np.deg2rad(40),
            ]

        e_N = self._x_var[:, N] - self._ref_p[:, N]
        cost += cp.sum_squares(P_chol @ e_N)
        self._prob = cp.Problem(cp.Minimize(cost), cons)
        self._first_solve = True   # trigger cold start after rebuild

    def _solve(self, x0: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Solve QP and return first input u[:,0]."""
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
            self._first_solve = False   # always warm-start on next call
        except Exception:
            self._first_solve = False
        return self._last_u.copy()

    # ── Main interface ────────────────────────────────────────────────────
    def compute(self, model, data, cmd: ControlCommand) -> np.ndarray:
        """
        Returns MuJoCo ctrl vector (6 elements):
          [T_bl, T_br, F_leg_L, F_leg_R, T_wl, T_wr]
        """
        if self._joint_ids is None:
            self._joint_ids = get_joint_ids(model)

        st = extract_state(model, data, self._joint_ids)
        x0 = state_to_vec(st)

        # Relinearize if leg lengths drifted (rebuilds CVXPY problem internally)
        self._relinearize(st["L_l"], st["L_r"])

        # Reference trajectory constant over horizon
        ref = np.zeros((10, self.N + 1))
        # Track s=0 to prevent slow drift (position reference held at 0)
        ref[1, :] = cmd.vx_ref
        ref[2, :] = cmd.yaw_ref

        u_mpc = self._solve(x0, ref)   # [T_wl, T_wr, T_bl, T_br] in MATLAB convention
        T_wl, T_wr, T_bl, T_br = u_mpc

        L_ref_l, L_ref_r = _compute_leg_refs(cmd.h_ref, st["theta_ll"], st["theta_lr"])

        F_l = _leg_pd(L_ref_l, st["L_l"], st["leg_L_dot"])
        F_r = _leg_pd(L_ref_r, st["L_r"], st["leg_R_dot"])

        # Sign conventions from brute-force 8s search (θll=-1, θb=+1, ds=-1, hip=+1, whl=-1):
        # ctrl_hip = T_bl (hip=+1, no negate)
        # ctrl_whl = -T_wl (whl=-1, negate)
        # ctrl order: [hip_L, hip_R, leg_L, leg_R, wheel_L, wheel_R]
        return np.array([T_bl, T_br, F_l, F_r, -T_wl, -T_wr])

    @property
    def last_u(self) -> np.ndarray:
        """Last MPC solution [T_wl, T_wr, T_bl, T_br] for logging."""
        return self._last_u.copy()
