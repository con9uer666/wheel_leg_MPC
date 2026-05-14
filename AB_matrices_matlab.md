# MATLAB-computed A, B matrices for the wheel-leg LQR/MPC model at L = 0.15 m

Source: `LQR.m` with the parameters defined at lines 100–128 (R_w=0.06, R_l=0.19242,
l_c=-0.01066729, m_w=0.615, m_l=2.47507, m_b=12.634, I_w=1.07156e-3,
I_b=0.30949668, I_z=0.62018662; leg-pole row at L=0.15 from `Leg_data_l`).

State vector x (10 entries):
    [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]

Input vector u (4 entries):
    [T_wl, T_wr, T_bl, T_br]

System: continuous-time, ẋ = A x + B u.

## A matrix (10×10)

```
0  1                0  0   0           0   0           0   0          0
0  0                0  0  -8.340622648 0  -8.340622648 0   0          0
0  0                0  1   0           0   0           0   0          0
0  0                0  0  -1.711719531 0   1.711719531 0   0          0
0  0                0  0   0           1   0           0   0          0
0  0                0  0  134.943988   0   6.986244483 0   0          0
0  0                0  0   0           0   0           1   0          0
0  0                0  0   6.986244483 0 134.943988    0   0          0
0  0                0  0   0           0   0           0   0          1
0  0                0  0   1.003341432 0   1.003341432 0  -4.271771237 0
```

## B matrix (10×4)

```
 0             0            0            0
 3.064235631   3.064235631 -0.7454056031 -0.7454056031
 0             0            0            0
-4.550763874   4.550763874 -0.1529772276  0.1529772276
 0             0            0            0
-35.31567108  -2.215562828 12.06001147    0.6243641514
 0             0            0            0
-2.215562828 -35.31567108   0.6243641514 12.06001147
 0             0            0            0
 0.1085983915  0.1085983915 -3.141383306 -3.141383306
```

## Notes

- A’s odd rows (1, 3, 5, 7, 9 in 1-indexed / 0, 2, 4, 6, 8 in 0-indexed) are the
  kinematic position→velocity links (single 1 on the super-diagonal).
- Gravity terms appear only in the even rows: A(2,5), A(2,7), A(4,5), A(4,7),
  A(6,5), A(6,7), A(8,5), A(8,7), A(10,5), A(10,7), A(10,9).
- Symmetry expected from left/right legs: A(6,5)=A(8,7), A(6,7)=A(8,5),
  B(2,:) symmetric, etc.  This is consistent with what we see.
- These matrices are what MATLAB `icare` actually saw when producing the K
  matrix in `新建 文本文档.txt`. Use them for any MPC design that needs to
  match the LQR-style behaviour.
