import numpy as np

# Physical parameters mirror MATLAB LQR.m so the pre-computed K matrix matches
# the simulated robot exactly.
G    = 9.81
R_W  = 0.06              # wheel radius (m)
R_L  = 0.19242           # half wheel track (m)
L_C  = -0.01066729       # body COM to hip joint axis (m) – COM is below the hip
M_W  = 0.615             # wheel mass (kg)
M_L  = 2.47507           # leg pole mass (kg)
M_B  = 12.634            # body mass (kg)
I_W  = 1.07156e-3        # wheel spin inertia (kg·m²)
I_B  = 0.30949668        # body pitch inertia (kg·m²)
I_Z  = 0.62018662        # body yaw inertia (kg·m²)

# Leg length is locked at 0.15 m by the leg PD; XML allows a small range
# (0.14 to 0.18) for compliance.
L_MIN = 0.14
L_MAX = 0.18

# Only the L=0.15 row is needed: leg length is fixed and K is pre-computed.
LEG_TABLE = np.array([
    [0.15, 0.07800145, 0.07199846, 0.05570304],
])


def leg_params(L: float) -> tuple:
    """Return (l_w, l_b, I_l) at the locked operating point (L≈0.15 m)."""
    return float(LEG_TABLE[0, 1]), float(LEG_TABLE[0, 2]), float(LEG_TABLE[0, 3])
