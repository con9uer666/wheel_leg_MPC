# MATLAB-computed A, B matrices for the 14-state wheel-leg MPC model at L = 0.15 m

Source: `LQR_14state.m` with parameters from LQR.m (R_w=0.06, R_l=0.19242,
l_c=-0.01066729, m_w=0.615, m_l=2.47507, m_b=12.634, I_w=1.07156e-3,
I_b=0.30949668, I_z=0.62018662) plus leg slide damping damping_L=5.0
(matches MuJoCo XML).

State vector x (14 entries):
    [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr,
     theta_b, dtheta_b, L_l, dL_l, L_r, dL_r]

Input vector u (6 entries):
    [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]

System: continuous-time, ẋ = A x + B u.

## A matrix (14×14)

Top-left 10×10 block = the 10-state A matrix (unchanged from model10.py).
Rows 11–14 / cols 11–14 = decoupled L subsystem:
  - dL_l/dt   = +1 * dL_l         (row 11, col 12)
  - ddL_l/dt  = -2.020 * dL_l     (row 12, col 12)  [= -damping/m_l]
  - dL_r/dt   = +1 * dL_r         (row 13, col 14)
  - ddL_r/dt  = -2.020 * dL_r     (row 14, col 14)

No off-diagonal coupling between L subsystem and the 10-state body
dynamics — this is the consequence of small-angle linearization at
theta_ll = theta_lr = 0.

```
A = [
 row  0  1     0  0  0  0   0      0   0       0   0   0       0   0
 row  0  0     0  0  0  0  -8.34   0  -8.34    0   0   0       0   0
 row  0  0     0  1  0  0   0      0   0       0   0   0       0   0
 row  0  0     0  0  0  0  -1.712  0   1.712   0   0   0       0   0
 row  0  0     0  0  0  1   0      0   0       0   0   0       0   0
 row  0  0     0  0  0  0  134.94  0   6.99    0   0   0       0   0
 row  0  0     0  0  0  0   0      1   0       0   0   0       0   0
 row  0  0     0  0  0  0   6.99   0  134.94   0   0   0       0   0
 row  0  0     0  0  0  0   0      0   0       1   0   0       0   0
 row  0  0     0  0  0  0   1.003  0   1.003   0  -4.272 0     0   0
 row  0  0     0  0  0  0   0      0   0       0   0   1       0   0
 row  0  0     0  0  0  0   0      0   0       0   0  -2.020   0   0
 row  0  0     0  0  0  0   0      0   0       0   0   0       0   1
 row  0  0     0  0  0  0   0      0   0       0   0   0       0  -2.020
]
```

## B matrix (14×6)

Cols 1-4 = original 10-state B (T_wl, T_wr, T_bl, T_br control) — only
affects rows 1-10.
Cols 5-6 = NEW F_leg_l, F_leg_r → each independently drives one leg:
  - F_leg_l affects only row 12 (ddL_l): B[12, 5] = +0.404 = 1/m_l
  - F_leg_r affects only row 14 (ddL_r): B[14, 6] = +0.404 = 1/m_l

```
B = [
   0        0       0       0       0      0
   3.0642   3.0642 -0.7454 -0.7454  0      0
   0        0       0       0       0      0
  -4.5508   4.5508 -0.1530  0.1530  0      0
   0        0       0       0       0      0
 -35.3157  -2.2156 12.0600  0.6244  0      0
   0        0       0       0       0      0
  -2.2156 -35.3157  0.6244 12.0600  0      0
   0        0       0       0       0      0
   0.1086   0.1086 -3.1414 -3.1414  0      0
   0        0       0       0       0      0
   0        0       0       0       0.404  0
   0        0       0       0       0      0
   0        0       0       0       0      0.404
]
```

## Verification

The 14-state model decomposes cleanly:
  - Top-left 10×10 of A14 ≡ A10 from model10.py (after the sign fixes)
  - Top-left 10×4 of B14 ≡ B10 (same 4 wheel/hip torque columns)
  - L subsystem (rows/cols 11-14, B cols 5-6) is a pure 2nd-order linear
    spring-damper with no spring (F_leg is the only force; damping_L=5 / m_l=2.47 ≈ 2.02 /s)

This means: in the linearised model, MPC controlling L through F_leg is
mathematically equivalent to the current PD law (modulo gains). The
benefit over PD shows up only when:
  - Reference L_ref(t) is a fast trajectory (MPC's preview compensates lag)
  - System leaves the linearization point (then theta_ll affects ddL via
    cos(theta_ll), which is dropped here — but this is also dropped from
    the existing PD anyway, so they're on equal footing)
