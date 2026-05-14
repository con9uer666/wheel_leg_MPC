"""
MPC controller for the 10-state wheel-leg robot.

Drop-in replacement for LQR10Controller sharing the exact same:
  - Observation vector (10 states): [s, ds, phi, dphi, theta_ll, dtheta_ll,
                                     theta_lr, dtheta_lr, theta_b, dtheta_b]
  - Control vector  (4 inputs):     [T_wl, T_wr, T_bl, T_br]
  - Sign conventions (mirrored from lqr10.py)
  - Leg PD law      (K_P=2500, K_D=120, locked at L=0.15 m)
  - Linearization   (get_AB10 + discretize from controllers.model10)

The controller solves a 20-step QP at 500 Hz (DT=0.002 s, matching the physics
step) using the OSQP solver via its native Python API. CVXPY was originally
used but its warm-start overhead made the per-step latency ~11 ms, which is
too slow for a system with a +15.6 unstable mode (~64 ms time constant).
The native OSQP build below converges to eps=1e-3 in ~125 iters at <1.5 ms
typical, leaving headroom for 500 Hz operation.

Output MuJoCo ctrl: [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import osqp

from .params import R_W, L_MIN, L_MAX
from .model10 import get_AB10, discretize
from sim.state import extract_state, get_joint_ids
from sim.command import ControlCommand
from config import CFG


# ── Sign conventions (loaded from config.yaml → signs section) ────────────
_SIGN_S        = int(CFG.signs.S)
_SIGN_PHI      = int(CFG.signs.PHI)
_SIGN_THETA_LL = int(CFG.signs.THETA_LL)
_SIGN_THETA_B  = int(CFG.signs.THETA_B)
_SIGN_HIP_OUT  = int(CFG.signs.HIP_OUT)
_SIGN_WHL_OUT  = int(CFG.signs.WHL_OUT)

# ── Leg PD gains (loaded from config.yaml → leg_pd section) ──────────────
_K_LEG_P  = float(CFG.leg_pd.K_P)
_K_LEG_D  = float(CFG.leg_pd.K_D)
_LEG_FF   = float(CFG.leg_pd.FF)


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
    """20-step linear MPC at 500 Hz, locked at L=0.15 m leg length.

    All tunable parameters (Q, R, N, DT, LEG_LEN, U_MAX, OSQP settings) are
    loaded from config.yaml at import time. See controllers/params.py for
    physical robot constants.
    """

    N       = int(CFG.mpc.N)
    DT      = float(CFG.mpc.DT)
    LEG_LEN = float(CFG.mpc.LEG_LEN)
    Q       = np.diag(CFG.mpc.Q)
    R       = np.diag(CFG.mpc.R)
    U_MAX   = CFG.actuator.U_MAX.copy()    # [T_wl, T_wr, T_bl, T_br]
    NX = 10
    NU = 4

    def __init__(self):
        # ── Linearize once at the locked leg length ──────────────────────
        A_c, B_c = get_AB10(self.LEG_LEN, self.LEG_LEN)
        self.Ad, self.Bd = discretize(A_c, B_c, self.DT)
        self.P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

        # ── Build the OSQP problem (constant H, C; b updated per solve) ──
        self._build_osqp()

        # ── Cached state ────────────────────────────────────────────────
        self._joint_ids = None
        self._last_u    = np.zeros(self.NU)
        self._n_fail    = 0
        self._n_solve   = 0

    # ── QP build (native OSQP, dense via scipy.sparse) ───────────────────
    def _build_osqp(self):
        N, NX, NU = self.N, self.NX, self.NU
        n_z = (N + 1) * NX + N * NU            # decision vector

        # Cost: 0.5 z' H z + g' z;  g = 0 baseline (ref offset added via x0 only)
        H = sp.block_diag(
            [sp.csc_matrix(self.Q)] * N
            + [sp.csc_matrix(self.P)]
            + [sp.csc_matrix(self.R)] * N,
            format='csc',
        )
        self._H = H
        # g will be set per solve to encode the reference (linear term from ref'Q*x)
        self._g_template = np.zeros(n_z)

        # Constraints: equality (dynamics + initial), then box on u
        rows, cols, vals = [], [], []
        # x_0 = x_init  (set in update)
        for i in range(NX):
            rows.append(i); cols.append(i); vals.append(1.0)
        nr = NX

        # Dynamics: x_{k+1} - Ad x_k - Bd u_k = 0
        for k in range(N):
            for i in range(NX):
                rows.append(nr); cols.append((k+1)*NX + i); vals.append(1.0)
                for j in range(NX):
                    if self.Ad[i, j] != 0:
                        rows.append(nr); cols.append(k*NX + j); vals.append(-self.Ad[i, j])
                for j in range(NU):
                    if self.Bd[i, j] != 0:
                        rows.append(nr); cols.append((N+1)*NX + k*NU + j); vals.append(-self.Bd[i, j])
                nr += 1

        # Input box constraints
        for k in range(N):
            for j in range(NU):
                rows.append(nr); cols.append((N+1)*NX + k*NU + j); vals.append(1.0)
                nr += 1

        C = sp.csc_matrix((vals, (rows, cols)), shape=(nr, n_z))
        self._C = C

        l = np.zeros(nr)
        u = np.zeros(nr)
        # Initial state row + dynamics equalities default to 0
        # Input bounds:
        for k in range(N):
            for j in range(NU):
                idx = NX + N * NX + k * NU + j
                l[idx] = -self.U_MAX[j]
                u[idx] = +self.U_MAX[j]
        self._l = l
        self._u = u
        self._n_z = n_z
        self._n_dyn = NX + N * NX  # equality rows

        # Setup OSQP using settings from config.yaml → osqp section.
        self._osqp = osqp.OSQP()
        self._osqp.setup(
            self._H, self._g_template, self._C, self._l, self._u,
            verbose=False,
            max_iter=int(CFG.osqp.max_iter),
            eps_abs=float(CFG.osqp.eps_abs),
            eps_rel=float(CFG.osqp.eps_rel),
            polish=bool(CFG.osqp.polish),
            adaptive_rho=bool(CFG.osqp.adaptive_rho),
            rho=float(CFG.osqp.rho),
            warm_starting=bool(CFG.osqp.warm_start),
        )

    def _solve(self, x0: np.ndarray, vx_ref: float, yaw_ref: float) -> np.ndarray:
        """Update bounds + linear cost, solve, return u_0."""
        self._n_solve += 1
        N, NX, NU = self.N, self.NX, self.NU

        # 1) Initial-state equality: l[0:NX] = u[0:NX] = x0
        self._l[:NX] = x0
        self._u[:NX] = x0

        # 2) Linear cost term g = -Q' * ref (since 1/2 ||x - ref||²_Q = 1/2 x'Qx - ref'Qx + const)
        #    Only x[1] (ds) and x[2] (phi) have nonzero references in this control task.
        g = self._g_template.copy()
        # Build per-stage cost gradient: g_x = -Q @ ref_x
        ref = np.zeros(NX)
        ref[1] = vx_ref
        ref[2] = yaw_ref * _SIGN_PHI
        g_x_stage = -self.Q @ ref
        g_x_term  = -self.P @ ref
        for k in range(N):
            g[k*NX:(k+1)*NX] = g_x_stage
        g[N*NX:(N+1)*NX] = g_x_term

        # OSQP supports updating l, u, q (linear cost) without re-factorising
        self._osqp.update(q=g, l=self._l, u=self._u)
        result = self._osqp.solve()

        if result.info.status == 'solved' or result.info.status == 'solved inaccurate':
            # u_0 is at offset (N+1)*NX in the decision vector
            u0_start = (N + 1) * NX
            self._last_u = result.x[u0_start:u0_start + NU].copy()
        else:
            self._n_fail += 1
            # Reuse previous solution to avoid sudden zero-output

        return self._last_u.copy()

    # ── Main interface ───────────────────────────────────────────────────
    def compute(self, model, data, cmd: ControlCommand) -> np.ndarray:
        """Called every physics step (500 Hz). Returns MuJoCo ctrl vector."""
        if self._joint_ids is None:
            self._joint_ids = get_joint_ids(model)

        # ── Build x0 in LQR/MATLAB sign convention ────────────────────────
        st = extract_state(model, data, self._joint_ids)
        x0 = np.array([
            st["s"]        * _SIGN_S,        st["ds"]        * _SIGN_S,
            st["phi"]      * _SIGN_PHI,      st["dphi"]      * _SIGN_PHI,
            st["theta_ll"] * _SIGN_THETA_LL, st["dtheta_ll"] * _SIGN_THETA_LL,
            st["theta_lr"] * _SIGN_THETA_LL, st["dtheta_lr"] * _SIGN_THETA_LL,
            st["theta_b"]  * _SIGN_THETA_B,  st["dtheta_b"]  * _SIGN_THETA_B,
        ])

        # ── Solve QP and unpack ───────────────────────────────────────────
        u_mpc = self._solve(x0, cmd.vx_ref, cmd.yaw_ref)
        T_wl, T_wr, T_bl, T_br = u_mpc

        # Leg PD (identical to LQR)
        L_ref_l, L_ref_r = _compute_leg_refs(cmd.h_ref,
                                              st["theta_ll"], st["theta_lr"])
        F_l = _leg_pd(L_ref_l, st["L_l"], st["leg_L_dot"])
        F_r = _leg_pd(L_ref_r, st["L_r"], st["leg_R_dot"])

        return np.array([
            T_bl * _SIGN_HIP_OUT,
            T_br * _SIGN_HIP_OUT,
            F_l, F_r,
            T_wl * _SIGN_WHL_OUT,
            T_wr * _SIGN_WHL_OUT,
        ])

    @property
    def last_u(self) -> np.ndarray:
        """Last solved [T_wl, T_wr, T_bl, T_br] — for runner logging."""
        return self._last_u.copy()
