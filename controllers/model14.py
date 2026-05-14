"""
14-state wheel-leg dynamics model (extends model10.py to include leg-length
dynamics).

This module produces the continuous-time A (14×14) and B (14×6) matrices
that MATLAB LQR_14state.m derives symbolically.  The user verified the
matrices match to 1e-8 by running the MATLAB script — see
AB_matrices_14state_matlab.md for the reference numbers.

State vector (14):
    x = [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr,
         theta_b, dtheta_b, L_l, dL_l, L_r, dL_r]

Input vector (6):
    u = [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]

Why the structure is block-diagonal:
    In the linearised model at theta_ll = theta_lr = 0, the leg-length
    dynamics decouples from the body dynamics:
        m_l * ddL = F_leg - damping_L * dL - m_l * g
    The gravity term is a constant bias that doesn't enter A or B; the
    damping_L term gives A[12, 12] = -damping_L / m_l ≈ -2.02; and
    B[12, 5] = +1 / m_l ≈ +0.404 (likewise for the right leg).
    The whole A and B match the existing 10-state model in the top-left
    block, with the new L rows/columns appended.
"""

import numpy as np

from .params import G, R_W, R_L, L_C, M_W, M_L, M_B, I_W, I_B, I_Z, leg_params
from .model10 import _compute_AB as _compute_AB10


# ── Leg slide damping (must match MuJoCo XML and LQR_14state.m) ──────────
# wheel_legged.xml: <joint name="leg_L" ... damping="5.0"/>
# LQR_14state.m:    damping_L_ac = 5.0
DAMPING_L = 5.0


def get_AB14(L_l: float, L_r: float):
    """Return continuous-time (A_14×14, B_14×6) at given leg lengths.

    The top-left 10×10 block of A and 10×4 block of B are EXACTLY the
    model10 outputs (re-uses _compute_AB from model10.py to guarantee
    consistency).  Rows 11-14 / cols 11-14 / cols 5-6 of B are filled in
    here from the closed-form leg-length dynamics.
    """
    # ── Reuse model10 for the 10-state body block ───────────────────────
    l_wl, l_bl, I_ll = leg_params(L_l)
    l_wr, l_br, I_lr = leg_params(L_r)
    A10, B10 = _compute_AB10(L_l, l_wl, l_bl, I_ll, L_r, l_wr, l_br, I_lr)

    # ── Allocate 14×14, 14×6 zeros ──────────────────────────────────────
    A = np.zeros((14, 14))
    B = np.zeros((14, 6))

    # Top-left blocks = 10-state model
    A[:10, :10] = A10
    B[:10, :4]  = B10

    # ── L subsystem (rows 10-13, 0-indexed) ─────────────────────────────
    # Row 10: dL_l = +1 * dL_l (state 11 in 0-index)
    # Row 11: ddL_l = -damping/m_l * dL_l + (1/m_l) * F_leg_l
    # Row 12: dL_r = +1 * dL_r (state 13)
    # Row 13: ddL_r = -damping/m_l * dL_r + (1/m_l) * F_leg_r
    A[10, 11] = 1.0
    A[11, 11] = -DAMPING_L / M_L
    A[12, 13] = 1.0
    A[13, 13] = -DAMPING_L / M_L

    B[11, 4] = 1.0 / M_L
    B[13, 5] = 1.0 / M_L

    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-order hold discretization via matrix exponential."""
    import scipy.linalg
    n, m = A.shape[0], B.shape[1]
    Z = np.zeros((n + m, n + m))
    Z[:n, :n] = A
    Z[:n, n:] = B
    eZ = scipy.linalg.expm(Z * dt)
    return eZ[:n, :n], eZ[:n, n:]


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    A, B = get_AB14(0.15, 0.15)
    print("A (14x14) =")
    print(A)
    print()
    print("B (14x6) =")
    print(B)
    print()
    print(f"Leg-subsystem A[11,11] = {A[11,11]:.6f}  (= -damping/m_l = -{DAMPING_L/M_L:.6f})")
    print(f"Leg-subsystem B[11,4]  = {B[11,4]:.6f}   (= +1/m_l = +{1.0/M_L:.6f})")
