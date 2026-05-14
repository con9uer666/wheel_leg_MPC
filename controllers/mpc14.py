"""
14-state MPC controller for the wheel-leg robot.

Extends MPC10Controller by treating leg length as a controllable state
(rather than a passively PD-tracked variable).  The MPC now jointly
optimises all 6 inputs:
    [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]

Observation vector (14):
    [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr,
     theta_b, dtheta_b, L_l, dL_l, L_r, dL_r]

Output MuJoCo ctrl: [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]
(same MuJoCo ctrl order as mpc10.py, but F_leg now comes from MPC instead
of a separate PD law)

All tunable parameters (Q, R, U_MAX, N, DT, OSQP settings) are loaded from
config.yaml at import time.  See controllers/params.py for physical robot
constants.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import osqp

from .params import R_W, L_MIN, L_MAX, M_L, G
from .model14 import get_AB14, discretize
from sim.state import extract_state, get_joint_ids
from sim.command import ControlCommand
from config import CFG


# ── Sign conventions (from config.yaml → signs section) ──────────────────
_SIGN_S        = int(CFG.signs.S)
_SIGN_PHI      = int(CFG.signs.PHI)
_SIGN_THETA_LL = int(CFG.signs.THETA_LL)
_SIGN_THETA_B  = int(CFG.signs.THETA_B)
_SIGN_HIP_OUT  = int(CFG.signs.HIP_OUT)
_SIGN_WHL_OUT  = int(CFG.signs.WHL_OUT)
_SIGN_LEG_OUT  = int(CFG.signs.LEG_OUT)
_SIGN_L        = int(CFG.signs.L)

# ── Gravity feed-forward for the leg-length input ────────────────────────
# The linearised model drops the constant m_l*g bias in the L dynamics, so
# the MPC's F_leg solution alone produces zero steady-state extension.
# Steady-state F_leg required to hold L=0.15 in MuJoCo: ~6.3 N (per leg).
# This is much less than m_l*g=24N because the leg's prismatic joint has
# damping=5.0 (passive support) and the ground contact takes most of the
# body weight via the wheel.
#
# Loaded from config.yaml so users can re-tune without touching code.
_F_LEG_FF = float(CFG.mpc14.F_LEG_FF) if hasattr(CFG.mpc14, "F_LEG_FF") else 6.3


class MPC14Controller:
    """20-step linear MPC at 500 Hz with 14 states and 6 inputs."""

    N       = int(CFG.mpc14.N)
    DT      = float(CFG.mpc14.DT)
    LEG_LEN = float(CFG.mpc14.LEG_LEN)
    Q       = np.diag(CFG.mpc14.Q)
    R       = np.diag(CFG.mpc14.R)
    U_MAX   = CFG.actuator.U_MAX_14.copy()    # 6-vector
    NX = 14
    NU = 6

    def __init__(self):
        # ── Linearize once at the locked leg length ──────────────────────
        A_c, B_c = get_AB14(self.LEG_LEN, self.LEG_LEN)
        self.Ad, self.Bd = discretize(A_c, B_c, self.DT)
        # Terminal cost = discrete-time ARE solution
        self.P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

        # ── Build the OSQP problem ───────────────────────────────────────
        self._build_osqp()

        # ── Cached state ────────────────────────────────────────────────
        self._joint_ids = None
        self._last_u    = np.zeros(self.NU)
        self._n_fail    = 0
        self._n_solve   = 0

    # ── QP build (native OSQP, sparse) ────────────────────────────────────
    def _build_osqp(self):
        N, NX, NU = self.N, self.NX, self.NU
        n_z = (N + 1) * NX + N * NU

        # Cost matrix H = block_diag(Q,...,Q, P, R,...,R)
        H = sp.block_diag(
            [sp.csc_matrix(self.Q)] * N
            + [sp.csc_matrix(self.P)]
            + [sp.csc_matrix(self.R)] * N,
            format='csc',
        )
        self._H = H
        self._g_template = np.zeros(n_z)

        rows, cols, vals = [], [], []

        # Initial state equality: x_0 = x_init
        for i in range(NX):
            rows.append(i); cols.append(i); vals.append(1.0)
        nr = NX

        # Dynamics equalities: x_{k+1} - Ad x_k - Bd u_k = 0
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
        for k in range(N):
            for j in range(NU):
                idx = NX + N * NX + k * NU + j
                l[idx] = -self.U_MAX[j]
                u[idx] = +self.U_MAX[j]
        self._l = l
        self._u = u
        self._n_z = n_z

        # Setup OSQP
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

    def _solve(self, x0: np.ndarray,
               vx_ref: float, yaw_ref: float,
               L_ref_l: float, L_ref_r: float) -> np.ndarray:
        """Update bounds + linear cost, solve, return u_0."""
        self._n_solve += 1
        N, NX, NU = self.N, self.NX, self.NU

        # 1) Initial-state equality
        self._l[:NX] = x0
        self._u[:NX] = x0

        # 2) Build per-stage reference and linear cost g = -Q ref (stage) / -P ref (term)
        ref = np.zeros(NX)
        ref[1]  = vx_ref                          # ds slot
        ref[2]  = yaw_ref * _SIGN_PHI             # yaw slot
        ref[10] = L_ref_l * _SIGN_L               # L_l slot
        ref[12] = L_ref_r * _SIGN_L               # L_r slot

        g = self._g_template.copy()
        g_x_stage = -self.Q @ ref
        g_x_term  = -self.P @ ref
        for k in range(N):
            g[k*NX:(k+1)*NX] = g_x_stage
        g[N*NX:(N+1)*NX] = g_x_term

        self._osqp.update(q=g, l=self._l, u=self._u)
        result = self._osqp.solve()

        if result.info.status in ('solved', 'solved inaccurate'):
            u0_start = (N + 1) * NX
            self._last_u = result.x[u0_start:u0_start + NU].copy()
        else:
            self._n_fail += 1
            # Keep previous solution (controller inertia)

        return self._last_u.copy()

    # ── Main interface ───────────────────────────────────────────────────
    def compute(self, model, data, cmd: ControlCommand) -> np.ndarray:
        """Called every physics step (500 Hz). Returns MuJoCo ctrl (6-vector)."""
        if self._joint_ids is None:
            self._joint_ids = get_joint_ids(model)

        st = extract_state(model, data, self._joint_ids)
        x0 = np.array([
            st["s"]        * _SIGN_S,        st["ds"]        * _SIGN_S,
            st["phi"]      * _SIGN_PHI,      st["dphi"]      * _SIGN_PHI,
            st["theta_ll"] * _SIGN_THETA_LL, st["dtheta_ll"] * _SIGN_THETA_LL,
            st["theta_lr"] * _SIGN_THETA_LL, st["dtheta_lr"] * _SIGN_THETA_LL,
            st["theta_b"]  * _SIGN_THETA_B,  st["dtheta_b"]  * _SIGN_THETA_B,
            st["L_l"]      * _SIGN_L,        st["leg_L_dot"] * _SIGN_L,
            st["L_r"]      * _SIGN_L,        st["leg_R_dot"] * _SIGN_L,
        ])

        u = self._solve(x0,
                        cmd.vx_ref, cmd.yaw_ref,
                        cmd.L_ref_l, cmd.L_ref_r)
        T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r = u

        # Apply gravity feed-forward to the leg-length inputs (see _F_LEG_FF)
        F_leg_l_total = F_leg_l + _F_LEG_FF
        F_leg_r_total = F_leg_r + _F_LEG_FF

        # MuJoCo ctrl order: [hip_L, hip_R, leg_L, leg_R, wheel_L, wheel_R]
        return np.array([
            T_bl * _SIGN_HIP_OUT,
            T_br * _SIGN_HIP_OUT,
            F_leg_l_total * _SIGN_LEG_OUT,
            F_leg_r_total * _SIGN_LEG_OUT,
            T_wl * _SIGN_WHL_OUT,
            T_wr * _SIGN_WHL_OUT,
        ])

    @property
    def last_u(self) -> np.ndarray:
        """Last solved [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]."""
        return self._last_u.copy()
