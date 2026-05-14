"""
LQR controller for the 10-state wheel-leg robot.

Uses the 4×10 K matrix pre-computed in MATLAB (LQR.m) at the locked operating
point L_l = L_r = 0.15 m, with the physical parameters mirrored in
controllers/params.py and models/wheel_legged.xml.

All tunable parameters (K matrix, sign conventions, leg PD gains, torque
limits) are loaded from config.yaml at import time. See controllers/params.py
for physical robot constants.

State:  x = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]
Input:  u = [T_wl, T_wr, T_bl, T_br]   (MATLAB convention)
Output: MuJoCo ctrl = [T_bl, T_br, F_leg_L, F_leg_R, -T_wl, -T_wr]
"""

import numpy as np

from .params import G, R_W, L_MIN, L_MAX, M_W, M_L, M_B
from sim.state import extract_state, get_joint_ids
from sim.command import ControlCommand
from config import CFG


# ── K matrix (loaded from config.yaml → lqr.K) ───────────────────────────
# Source: MATLAB LQR.m icare output for Q=diag(10,300,5000,1,5000,1,5000,1,25000,1)
# and R=diag(40,40,1,1) at L=0.15 m.
_K = CFG.lqr.K.copy()

# ── Actuator limits (loaded from config.yaml → actuator.U_MAX) ───────────
_U_MAX = CFG.actuator.U_MAX.copy()

# ── Leg PD gains (loaded from config.yaml → leg_pd) ──────────────────────
_K_LEG_P  = float(CFG.leg_pd.K_P)
_K_LEG_D  = float(CFG.leg_pd.K_D)
_LEG_FF   = float(CFG.leg_pd.FF)

# ── Sign conventions (loaded from config.yaml → signs) ───────────────────
_SIGN_S        = int(CFG.signs.S)
_SIGN_PHI      = int(CFG.signs.PHI)
_SIGN_THETA_LL = int(CFG.signs.THETA_LL)
_SIGN_THETA_B  = int(CFG.signs.THETA_B)
_SIGN_HIP_OUT  = int(CFG.signs.HIP_OUT)
_SIGN_WHL_OUT  = int(CFG.signs.WHL_OUT)


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
