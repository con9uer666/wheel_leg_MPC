"""Shared control command object written by the dashboard sliders, read by MPC."""

from dataclasses import dataclass, field


@dataclass
class ControlCommand:
    vx_ref:  float = 0.0   # desired forward velocity (m/s)
    h_ref:   float = 0.21  # desired body height above ground (m)
    yaw_ref: float = 0.0   # desired yaw angle (rad)
    L_ref_l: float = 0.15  # desired left-leg total length (m)  — used by MPC14
    L_ref_r: float = 0.15  # desired right-leg total length (m) — used by MPC14
