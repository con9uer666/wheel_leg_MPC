"""Shared control command object written by the dashboard sliders, read by MPC."""

from dataclasses import dataclass, field


@dataclass
class ControlCommand:
    vx_ref:  float = 0.0   # desired forward velocity (m/s)
    h_ref:   float = 0.27  # desired body height above ground (m)
    yaw_ref: float = 0.0   # desired yaw angle (rad)
