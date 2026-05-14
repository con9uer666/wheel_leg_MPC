"""
10-state wheel-leg dynamics model derived numerically from MATLAB equations
(HerKules_VOCAL_SJ_LQR_v4_with_data.m, equations 3.11-3.15).

Uses numpy linear algebra (no SymPy) for fast evaluation.

State:  x = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]
Input:  u = [T_wl, T_wr, T_bl, T_br]
"""

import numpy as np
import scipy.linalg

from .params import G, R_W, R_L, L_C, M_W, M_L, M_B, I_W, I_B, I_Z, leg_params


def _compute_AB(l_l, l_wl, l_bl, I_ll, l_r, l_wr, l_br, I_lr):
    """
    Build continuous-time A (10×10) and B (10×4) numerically.

    The 5-equation system (eqn1..eqn5) is LINEAR in the 5 unknown
    accelerations [dd_wl, dd_wr, dd_ll, dd_lr, dd_b].
    We write it as  M @ acc = f(state_angles, inputs)  and solve with numpy.

    Rather than symbolic differentiation we directly extract the coefficients
    of each theta_* and T_* from f (since f is linear in both).

    The MATLAB Jacobian rules produce A and B via:
      J_A[i,j] = partial acc[i] / partial state_angle[j]
      J_B[i,j] = partial acc[i] / partial input[j]
    which equal:  J_A = M^{-1} @ dRHS_dangle,  J_B = M^{-1} @ dRHS_dinput
    """
    g = G
    R_w = R_W
    R_l = R_L
    l_c = L_C
    m_w = M_W
    m_l = M_L
    m_b = M_B
    I_w = I_W
    I_b = I_B
    I_z = I_Z

    # ── Mass matrix M (5×5) ──────────────────────────────────────────────
    # Each row corresponds to one equation; columns to [dd_wl, dd_wr, dd_ll, dd_lr, dd_b]
    M = np.zeros((5, 5))

    # eqn1 (wheel_L dynamics)
    M[0, 0] = I_w * l_l / R_w + m_w * R_w * l_l + m_l * R_w * l_bl
    M[0, 2] = m_l * l_wl * l_bl - I_ll

    # eqn2 (wheel_R dynamics)
    M[1, 1] = I_w * l_r / R_w + m_w * R_w * l_r + m_l * R_w * l_br
    M[1, 3] = m_l * l_wr * l_br - I_lr

    # eqn3 (longitudinal)
    c3 = m_w * R_w**2 + I_w + m_l * R_w**2 + m_b * R_w**2 / 2
    M[2, 0] = -c3
    M[2, 1] = -c3
    M[2, 2] = -(m_l * R_w * l_wl + m_b * R_w * l_l / 2)
    M[2, 3] = -(m_l * R_w * l_wr + m_b * R_w * l_r / 2)

    # eqn4 (body pitch)
    c4 = m_w * R_w * l_c + I_w * l_c / R_w + m_l * R_w * l_c
    M[3, 0] = c4
    M[3, 1] = c4
    M[3, 2] = m_l * l_wl * l_c
    M[3, 3] = m_l * l_wr * l_c
    M[3, 4] = -I_b

    # eqn5 (yaw)
    c5a = I_z * R_w / (2 * R_l) + I_w * R_l / R_w
    M[4, 0] =  c5a
    M[4, 1] = -c5a
    M[4, 2] =  I_z * l_l / (2 * R_l)
    M[4, 3] = -I_z * l_r / (2 * R_l)

    # ── RHS dependence on state angles [theta_ll, theta_lr, theta_b] ─────
    # dRHS_dangle:  5×3  (row=equation, col=angle index)
    # Only gravity terms survive differentiation w.r.t. angles.
    dRHS_dangle = np.zeros((5, 3))
    # eqn1: gravity term +(m_l*l_wl + m_b*l_l/2)*g*theta_ll  → col 0
    dRHS_dangle[0, 0] = -(m_l * l_wl + m_b * l_l / 2) * g
    # eqn2: gravity term +(m_l*l_wr + m_b*l_r/2)*g*theta_lr → col 1
    dRHS_dangle[1, 1] = -(m_l * l_wr + m_b * l_r / 2) * g
    # eqn4: gravity term +m_b*g*l_c*theta_b → col 2
    dRHS_dangle[3, 2] = -m_b * g * l_c

    # ── RHS dependence on inputs [T_wl, T_wr, T_bl, T_br] ───────────────
    # dRHS_dinput:  5×4
    dRHS_dinput = np.zeros((5, 4))
    # eqn1: -T_bl coefficient (RHS sign: +T_bl → moves to RHS as -T_bl)
    # Original: ... + T_bl - T_wl*(1+l_l/R_w) = 0
    # => M@acc = -(T_bl - T_wl*(1+l_l/R_w))
    dRHS_dinput[0, 0] =  (1 + l_l / R_w)   # T_wl coefficient on RHS
    dRHS_dinput[0, 2] = -1.0                # T_bl coefficient on RHS
    # eqn2: similarly
    dRHS_dinput[1, 1] =  (1 + l_r / R_w)
    dRHS_dinput[1, 3] = -1.0
    # eqn3: +T_wl + T_wr on RHS  (moved to RHS: negative of the original eqn)
    # Original: ... + T_wl + T_wr = 0  => M@acc = T_wl + T_wr  (wrong sign?)
    # Careful: eqn3 = -(c3)*ddw_l - (c3)*ddw_r - ... + T_wl + T_wr = 0
    # => M[2,:] @ acc = T_wl + T_wr
    dRHS_dinput[2, 0] =  1.0
    dRHS_dinput[2, 1] =  1.0
    # eqn4: -(T_wl+T_wr)*l_c/R_w - (T_bl+T_br) → RHS
    dRHS_dinput[3, 0] =  l_c / R_w
    dRHS_dinput[3, 1] =  l_c / R_w
    dRHS_dinput[3, 2] =  1.0
    dRHS_dinput[3, 3] =  1.0
    # eqn5: -T_wl*R_l/R_w + T_wr*R_l/R_w → RHS
    dRHS_dinput[4, 0] =  R_l / R_w
    dRHS_dinput[4, 1] = -R_l / R_w

    # ── Solve for Jacobians ───────────────────────────────────────────────
    # J_A (5×3): d acc / d angle  =  M^{-1} @ dRHS_dangle
    # J_B (5×4): d acc / d input  =  M^{-1} @ dRHS_dinput
    J_A = np.linalg.solve(M, dRHS_dangle)   # 5×3
    J_B = np.linalg.solve(M, dRHS_dinput)   # 5×4

    # ── Build A (10×10) following MATLAB rules ────────────────────────────
    A = np.zeros((10, 10))

    # Odd rows (0-indexed: 0,2,4,6,8): velocity→position links
    for i in range(0, 10, 2):
        A[i, i + 1] = 1.0

    # State angle columns: theta_ll→4, theta_lr→6, theta_b→8
    # J_A columns: 0→theta_ll, 1→theta_lr, 2→theta_b
    for col_j, state_col in enumerate([4, 6, 8]):
        # Row 1: ds_dot = R_w/2*(dd_wl + dd_wr)
        A[1, state_col] = R_w / 2 * (J_A[0, col_j] + J_A[1, col_j])
        # Row 3: dphi_dot (yaw) – from kinematic relation
        A[3, state_col] = (
            R_w / (2 * R_l) * (J_A[0, col_j] - J_A[1, col_j])
            - l_l / (2 * R_l) * J_A[2, col_j]
            + l_r / (2 * R_l) * J_A[3, col_j]
        )
        # Row 5: dtheta_ll_dot
        A[5, state_col] = J_A[2, col_j]
        # Row 7: dtheta_lr_dot
        A[7, state_col] = J_A[3, col_j]
        # Row 9: dtheta_b_dot
        A[9, state_col] = J_A[4, col_j]

    # ── Build B (10×4) ────────────────────────────────────────────────────
    B = np.zeros((10, 4))
    for h in range(4):
        B[1, h] = R_w / 2 * (J_B[0, h] + J_B[1, h])
        B[3, h] = (
            R_w / (2 * R_l) * (J_B[0, h] - J_B[1, h])
            - l_l / (2 * R_l) * J_B[2, h]
            + l_r / (2 * R_l) * J_B[3, h]
        )
        B[5, h] = J_B[2, h]
        B[7, h] = J_B[3, h]
        B[9, h] = J_B[4, h]

    return A, B


def get_AB10(L_l: float, L_r: float):
    """Return continuous-time A (10×10), B (10×4) for given leg lengths."""
    l_wl, l_bl, I_ll = leg_params(L_l)
    l_wr, l_br, I_lr = leg_params(L_r)
    return _compute_AB(L_l, l_wl, l_bl, I_ll, L_r, l_wr, l_br, I_lr)


def discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-order hold discretization via matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = A
    Z[:n, n:] = B
    eZ = scipy.linalg.expm(Z * dt)
    return eZ[:n, :n], eZ[:n, n:]
