"""
Physical parameters of the wheel-leg robot.

This file is the ONE place to edit when the simulated robot changes (e.g.
swapping wheel size, redesigning legs, retuning the MuJoCo XML masses).
All values mirror MATLAB LQR.m so the K matrix in config.yaml and the
linearisation in controllers/model10.py stay consistent.

Tuning parameters (Q, R, leg-PD gains, OSQP settings, sign conventions)
live in config.yaml — see config.py for the loader. DO NOT mix physical
constants and controller tuning in the same file.
"""

import numpy as np

# ── Gravity ──────────────────────────────────────────────────────────────
G    = 9.81                # m/s²

# ── Body / wheel masses and inertias (from MATLAB LQR.m) ──────────────────
R_W  = 0.06              # wheel radius (m)
R_L  = 0.19242           # half wheel track (m)
L_C  = -0.01066729       # body COM to hip joint axis (m) – COM is below the hip
M_W  = 0.615             # wheel mass (kg)
M_L  = 2.47507           # leg pole mass (kg)
M_B  = 12.634            # body mass (kg)
I_W  = 1.07156e-3        # wheel spin inertia (kg·m²)
I_B  = 0.30949668        # body pitch inertia (kg·m²)
I_Z  = 0.62018662        # body yaw inertia (kg·m²)

# ── Leg-length range ─────────────────────────────────────────────────────
# Leg length is locked at 0.15 m by the leg PD; XML allows a small range
# (0.14 to 0.18) for compliance.
L_MIN = 0.14
L_MAX = 0.18

# ── Leg-pole inertial table (l_w, l_b, I_l) vs total leg length L ────────
# Only the L=0.15 row is needed: leg length is fixed and the K matrix is
# pre-computed in MATLAB at exactly this operating point.
LEG_TABLE = np.array([
    [0.15, 0.07800145, 0.07199846, 0.05570304],
])


def leg_params(L: float) -> tuple:
    """Return (l_w, l_b, I_l) at the locked operating point (L≈0.15 m)."""
    return float(LEG_TABLE[0, 1]), float(LEG_TABLE[0, 2]), float(LEG_TABLE[0, 3])
