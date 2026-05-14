"""
LQR controller for the 10-state wheel-leg robot.

Uses the 4×10 K matrix pre-computed in MATLAB (LQR.m) at the locked operating
point L_l = L_r = 0.15 m, with the physical parameters mirrored in
controllers/params.py and models/wheel_legged.xml.

State:  x = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]
Input:  u = [T_wl, T_wr, T_bl, T_br]   (MATLAB convention)
Output: MuJoCo ctrl = [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]
"""

import numpy as np

from .params import G, R_W, L_MIN, L_MAX, M_W, M_L, M_B
from sim.state import extract_state, get_joint_ids
from sim.command import ControlCommand


# ── K matrix from MATLAB LQR.m at L=0.15 m ───────────────────────────────
# Source: 新建 文本文档.txt (user-provided icare output with MATLAB LQR.m's
# Q = diag(10, 300, 5000, 1, 5000, 1, 5000, 1, 25000, 1)
# R = diag(40, 40, 1, 1)).
# Note: scipy.linalg.solve_continuous_are on the same A,B,Q,R produces a
# different K (different sign convention / column scaling) that does not
# transfer back to MuJoCo with the signs below — see tune_lqr.py for a
# search harness that tunes Q/R in scipy-space if desired.
# Rows: [T_wl, T_wr, T_bl, T_br]; columns: the 10-state vector.
_K = np.array([
    [-0.22494, -1.4517, -7.1367, -1.1802, -8.9838, -0.63815, -4.1141, -0.42777, -12.874, -1.4493],
    [-0.22494, -1.4517,  7.1367,  1.1802, -4.1141, -0.42777, -8.9838, -0.63815, -12.874, -1.4493],
    [ 1.7251,  10.998,  -21.51,  -3.7978, 67.911,   3.8477,  -7.9052,  0.33238, -74.067, -2.5312 ],
    [ 1.7251,  10.998,   21.51,   3.7978, -7.9052,  0.33238, 67.911,   3.8477, -74.067, -2.5312 ],
])

# ── Actuator limits (must match wheel_legged.xml) ─────────────────────────
_U_MAX = np.array([30.0, 30.0, 15.0, 15.0])    # [T_wl, T_wr, T_bl, T_br]

# ── Leg PD gains ──────────────────────────────────────────────────────────
# Conservative gains (matching MPC10 baseline). Higher K_P (>5000) gives
# tighter leg-length tracking at rest but reduces robustness to disturbances
# — pushes >25 N flip the robot. Keeping it soft trades 1 cm of static height
# error for surviving ±100 N pushes.
_K_LEG_P  = 2500.0
_K_LEG_D  = 120.0
_LEG_FF   = 0.0

# ── Sign conventions ──────────────────────────────────────────────────────
# Grid-searched over MuJoCo (after fixing extract_state to return ds with the
# natural MuJoCo sign). Verified stable in both balance and 5° yaw-init tests.
# Two equivalent solutions exist (global negation); this one keeps hip_out=+1
# and wheel_out=-1 so the ctrl flow matches typical RoboMaster wheel-leg conventions.
_SIGN_S        = +1
_SIGN_PHI      = -1
_SIGN_THETA_LL = +1
_SIGN_THETA_B  = -1
_SIGN_HIP_OUT  = +1   # ctrl_hip_L = +T_bl
_SIGN_WHL_OUT  = -1   # ctrl_wheel_L = -T_wl


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


class LQR10Controller:
    """Fixed-gain LQR for the wheel-leg robot with locked leg length 0.15 m."""

    K     = _K
    U_MAX = _U_MAX

    def __init__(self):
        self._joint_ids = None
        self._last_u    = np.zeros(4)

    def compute(self, model, data, cmd: ControlCommand) -> np.ndarray:
        """Return MuJoCo ctrl: [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]."""
        if self._joint_ids is None:
            self._joint_ids = get_joint_ids(model)

        st = extract_state(model, data, self._joint_ids)

        x = np.array([
            st["s"]        * _SIGN_S,               st["ds"]        * _SIGN_S,
            st["phi"]      * _SIGN_PHI,             st["dphi"]      * _SIGN_PHI,
            st["theta_ll"] * _SIGN_THETA_LL,        st["dtheta_ll"] * _SIGN_THETA_LL,
            st["theta_lr"] * _SIGN_THETA_LL,        st["dtheta_lr"] * _SIGN_THETA_LL,
            st["theta_b"]  * _SIGN_THETA_B,         st["dtheta_b"]  * _SIGN_THETA_B,
        ])

        # Reference targets in MATLAB state space. ds-slot: extract_state already
        # returns -qvel[0], so x[1] = +qvel[0] (with _SIGN_S=-1) and a forward
        # vx_ref maps directly. phi-slot: extract_state returns +yaw, so we apply
        # _SIGN_PHI to convert MuJoCo yaw_ref into the state-space slot.
        x_ref     = np.zeros(10)
        x_ref[1]  = cmd.vx_ref
        x_ref[2]  = cmd.yaw_ref * _SIGN_PHI

        u = -self.K @ (x - x_ref)
        u = np.clip(u, -self.U_MAX, self.U_MAX)
        self._last_u = u.copy()
        T_wl, T_wr, T_bl, T_br = u

        L_ref_l, L_ref_r = _compute_leg_refs(cmd.h_ref,
                                              st["theta_ll"], st["theta_lr"])
        F_l = _leg_pd(L_ref_l, st["L_l"], st["leg_L_dot"])
        F_r = _leg_pd(L_ref_r, st["L_r"], st["leg_R_dot"])

        return np.array([
            T_bl * _SIGN_HIP_OUT,
            T_br * _SIGN_HIP_OUT,
            F_l,
            F_r,
            T_wl * _SIGN_WHL_OUT,
            T_wr * _SIGN_WHL_OUT,
        ])

    @property
    def last_u(self) -> np.ndarray:
        """Last solution [T_wl, T_wr, T_bl, T_br] for logging."""
        return self._last_u.copy()
