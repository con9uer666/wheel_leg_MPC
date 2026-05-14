#!/usr/bin/env python3
"""
Auto-tune LQR Q/R matrices against a fixed disturbance scenario.

Procedure
---------
1. For each Q/R candidate, compute K via scipy.linalg.solve_continuous_are
   on the model10 linearization at L=0.15 m.
2. Run a headless MuJoCo simulation:
     - 1 s settle to balance
     - 50 N forward push on trunk for 0.2 s
     - 4 s recovery window
3. Score = max |pitch| + 5 * |final_pitch| + 0.5 * |final_x| (all in rad, m)
   Lower is better. Any run that exceeds 30° pitch is rejected outright.
4. Signs are pinned to the ones validated for the current scipy-style K
   (auto-determined once at startup by a 64-combo MuJoCo grid search).
"""

import os
import sys
import time
import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco

from controllers.model10 import get_AB10
from controllers.params import L_MIN
from sim.state import get_joint_ids, extract_state, pitch_from_xmat

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "wheel_legged.xml")

# Disturbance: 50 N forward push on trunk between t=1.0 s and t=1.2 s
DISTURB_FORCE   = 50.0
DISTURB_T_START = 1.0
DISTURB_T_END   = 1.2

DT_SIM  = 0.002
DT_CTRL = 0.002    # 500 Hz LQR

U_MAX = np.array([30.0, 30.0, 15.0, 15.0])
K_LEG_P = 7000.0
K_LEG_D = 300.0
LEG_FF  = 70.0


# ─────────────────────────────────────────────────────────────────────────
def compute_K(Q, R, L=0.15):
    A, B = get_AB10(L, L)
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


def leg_pd(L_ref, L_meas, L_dot):
    return K_LEG_P * (L_ref - L_meas) - K_LEG_D * L_dot + LEG_FF


def run_sim(K, signs, duration=5.0, disturb=True, vx_ref=0.0):
    """
    signs = (s, phi, tll, tb, hip_out, whl_out) ∈ {-1, +1}^6
    Returns (max_pitch_deg, final_pitch_deg, final_x_m, crashed_bool, max_used_torque)
    """
    s_s, s_phi, s_tll, s_tb, s_ho, s_wo = signs

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    data.qpos[2] = 0.215
    mujoco.mj_forward(model, data)
    jids = get_joint_ids(model)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    t, t_last_ctrl = 0.0, -DT_CTRL
    ctrl = np.zeros(6)
    max_pitch_rad = 0.0
    crashed = False
    max_torque = 0.0

    while t < duration:
        # ── disturbance ─────────────────────────────────────────────
        if disturb and DISTURB_T_START <= t < DISTURB_T_END:
            data.xfrc_applied[trunk_id] = [DISTURB_FORCE, 0, 0, 0, 0, 0]
        else:
            data.xfrc_applied[trunk_id] = [0, 0, 0, 0, 0, 0]

        # ── control ─────────────────────────────────────────────────
        if t - t_last_ctrl >= DT_CTRL - 1e-9:
            t_last_ctrl = t
            st = extract_state(model, data, jids)

            x = np.array([
                st["s"]        * s_s,    st["ds"]       * s_s,
                st["phi"]      * s_phi,  st["dphi"]     * s_phi,
                st["theta_ll"] * s_tll,  st["dtheta_ll"]* s_tll,
                st["theta_lr"] * s_tll,  st["dtheta_lr"]* s_tll,
                st["theta_b"]  * s_tb,   st["dtheta_b"] * s_tb,
            ])
            x_ref = np.zeros(10)
            x_ref[1] = vx_ref

            u = -K @ (x - x_ref)
            u = np.clip(u, -U_MAX, U_MAX)
            T_wl, T_wr, T_bl, T_br = u
            max_torque = max(max_torque, float(np.abs(u).max()))

            F_l = leg_pd(0.15, st["L_l"], st["leg_L_dot"])
            F_r = leg_pd(0.15, st["L_r"], st["leg_R_dot"])
            ctrl = np.array([
                T_bl * s_ho, T_br * s_ho, F_l, F_r,
                T_wl * s_wo, T_wr * s_wo,
            ])

            max_pitch_rad = max(max_pitch_rad, abs(st["theta_b"]))
            if abs(st["theta_b"]) > np.deg2rad(30):
                crashed = True
                break

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        t += DT_SIM

    final_pitch = pitch_from_xmat(data.xmat[trunk_id])
    final_x     = float(data.qpos[0])
    return (np.degrees(max_pitch_rad), np.degrees(final_pitch),
            final_x, crashed, max_torque)


def score_run(max_pitch, final_pitch, final_x, crashed):
    """Lower is better. Crash = inf."""
    if crashed:
        return float("inf")
    return max_pitch + 5.0 * abs(final_pitch) + 50.0 * abs(final_x)


# ─────────────────────────────────────────────────────────────────────────
# Step 1: find sign convention that works for scipy-style K
# ─────────────────────────────────────────────────────────────────────────
def run_sim_with_init(K, signs, init_yaw=0.0, init_pitch=0.0, duration=15.0):
    """Like run_sim but injects an initial perturbation to flush slow modes."""
    s_s, s_phi, s_tll, s_tb, s_ho, s_wo = signs

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    data.qpos[2] = 0.215
    # Set initial yaw via quaternion (rotation about +z)
    if init_yaw != 0.0:
        data.qpos[3] = np.cos(init_yaw / 2)
        data.qpos[6] = np.sin(init_yaw / 2)
    mujoco.mj_forward(model, data)
    jids = get_joint_ids(model)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    t, t_last_ctrl = 0.0, -DT_CTRL
    ctrl = np.zeros(6)
    max_pitch_rad = 0.0
    max_yaw_rad   = abs(init_yaw)
    crashed = False

    while t < duration:
        if t - t_last_ctrl >= DT_CTRL - 1e-9:
            t_last_ctrl = t
            st = extract_state(model, data, jids)
            x = np.array([
                st["s"]        * s_s,    st["ds"]       * s_s,
                st["phi"]      * s_phi,  st["dphi"]     * s_phi,
                st["theta_ll"] * s_tll,  st["dtheta_ll"]* s_tll,
                st["theta_lr"] * s_tll,  st["dtheta_lr"]* s_tll,
                st["theta_b"]  * s_tb,   st["dtheta_b"] * s_tb,
            ])
            u = -K @ x
            u = np.clip(u, -U_MAX, U_MAX)
            T_wl, T_wr, T_bl, T_br = u
            F_l = leg_pd(0.15, st["L_l"], st["leg_L_dot"])
            F_r = leg_pd(0.15, st["L_r"], st["leg_R_dot"])
            ctrl = np.array([
                T_bl * s_ho, T_br * s_ho, F_l, F_r,
                T_wl * s_wo, T_wr * s_wo,
            ])
            max_pitch_rad = max(max_pitch_rad, abs(st["theta_b"]))
            max_yaw_rad   = max(max_yaw_rad,   abs(st["phi"]))
            if abs(st["theta_b"]) > np.deg2rad(30) or abs(st["phi"]) > np.deg2rad(60):
                crashed = True
                break
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        t += DT_SIM

    final_yaw = float(extract_state(model, data, jids)["phi"])
    return (np.degrees(max_pitch_rad), np.degrees(max_yaw_rad),
            np.degrees(final_yaw), crashed)


def find_signs(K):
    """Grid-search 64 sign combos; tests both balance recovery and yaw stability."""
    best = (float("inf"), None)
    for s_s in (-1, 1):
        for s_phi in (-1, 1):
            for s_tll in (-1, 1):
                for s_tb in (-1, 1):
                    for s_ho in (-1, 1):
                        for s_wo in (-1, 1):
                            signs = (s_s, s_phi, s_tll, s_tb, s_ho, s_wo)
                            # Test 1: balance recovery from yaw perturbation
                            mp, my, fy, cr = run_sim_with_init(
                                K, signs, init_yaw=np.deg2rad(5), duration=15.0)
                            if cr:
                                continue
                            # Score: peak pitch + final yaw drift heavily penalised
                            sc = mp + 0.3*my + 2.0*abs(fy)
                            if sc < best[0]:
                                best = (sc, signs, (mp, my, fy))
    return best[1] if best[1] is not None else None


# ─────────────────────────────────────────────────────────────────────────
# Step 2: search Q/R
# ─────────────────────────────────────────────────────────────────────────
BASELINE_Q = np.array([10, 300, 5000, 1, 5000, 1, 5000, 1, 25000, 1], dtype=float)
BASELINE_R = np.array([40, 40, 1, 1], dtype=float)


def sample_QR(rng):
    """Sample log-uniform variations of the baseline Q/R."""
    Q = BASELINE_Q.copy()
    # Tweak position weights independently
    Q[0]  *= 10 ** rng.uniform(-0.5, 1.5)    # s
    Q[1]  *= 10 ** rng.uniform(-0.5, 1.5)    # ds
    Q[2]  *= 10 ** rng.uniform(-1.0, 1.0)    # phi
    Q[3]  *= 10 ** rng.uniform(-0.5, 2.0)    # dphi
    Q[4]  *= 10 ** rng.uniform(-1.0, 1.0)    # theta_ll
    Q[6]  = Q[4]                              # symmetry
    Q[5]  *= 10 ** rng.uniform(-0.5, 2.0)    # dtheta_ll
    Q[7]  = Q[5]
    Q[8]  *= 10 ** rng.uniform(-0.5, 1.0)    # theta_b
    Q[9]  *= 10 ** rng.uniform(-0.5, 2.5)    # dtheta_b

    R = BASELINE_R.copy()
    R[0]  *= 10 ** rng.uniform(-2.0, 0.5)    # T_wl  (try smaller for faster wheel response)
    R[1]  = R[0]
    R[2]  *= 10 ** rng.uniform(-0.5, 0.5)    # T_bl
    R[3]  = R[2]
    return np.diag(Q), np.diag(R)


def main():
    rng = np.random.default_rng(42)

    # ── Use the user K's empirically-verified sign convention ─────────────
    # (Found previously to give perfect yaw tracking with MATLAB-format K.)
    signs = (-1, -1, +1, -1, +1, -1)
    print(f"Using fixed signs (validated for yaw stability):")
    print(f"  (s={signs[0]:+d}, phi={signs[1]:+d}, tll={signs[2]:+d}, "
          f"tb={signs[3]:+d}, hip={signs[4]:+d}, whl={signs[5]:+d})")

    # Quick check: does the baseline scipy K + these signs actually balance?
    K0 = compute_K(np.diag(BASELINE_Q), np.diag(BASELINE_R))
    mp, my, fy, cr = run_sim_with_init(K0, signs, init_yaw=np.deg2rad(5), duration=15.0)
    if cr:
        print(f"  Baseline scipy K + these signs CRASHES "
              f"(max_pitch={mp:.1f}°, max_yaw={my:.1f}°)")
        print("  Falling back to sign search …")
        signs = find_signs(K0)
        if signs is None:
            print("  No stable sign combo found!"); return
        print(f"  search-found signs = {signs}")
    else:
        print(f"  baseline K + signs:  max_pitch={mp:.2f}°  "
              f"max_yaw={my:.2f}°  final_yaw={fy:.3f}°")

    base_res = run_sim(K0, signs, duration=5.0, disturb=True)
    print(f"  baseline disturb run: max_pitch={base_res[0]:.2f}°  "
          f"final_pitch={base_res[1]:.3f}°  final_x={base_res[2]:+.3f}m  "
          f"crashed={base_res[3]}  score={score_run(*base_res[:4]):.3f}")

    # ── search ───────────────────────────────────────────────────────────────
    N_TRIALS = 60
    print(f"\nStep 2: searching {N_TRIALS} Q/R combos …")
    results = []
    for i in range(N_TRIALS):
        Q, R = sample_QR(rng)
        try:
            K = compute_K(Q, R)
        except Exception:
            continue
        res = run_sim(K, signs, duration=5.0, disturb=True)
        sc  = score_run(*res[:4])
        results.append((sc, Q, R, K, res))
        if i % 10 == 9 or i == N_TRIALS - 1:
            print(f"  [{i+1:2d}/{N_TRIALS}] best so far: "
                  f"{min(r[0] for r in results):.3f}")

    results.sort(key=lambda r: r[0])
    print("\nTop 5 candidates:")
    for k, (sc, Q, R, K, res) in enumerate(results[:5]):
        print(f"  #{k+1}: score={sc:.3f}  "
              f"max_pitch={res[0]:.2f}°  final_pitch={res[1]:.3f}°  "
              f"final_x={res[2]:+.3f}m  max_torque={res[4]:.1f}")

    best = results[0]
    print(f"\nBest: score={best[0]:.3f}")
    print(f"  Q diag: {np.diag(best[1])}")
    print(f"  R diag: {np.diag(best[2])}")

    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    print("\nOptimal K:")
    print(best[3])

    # Save the best K and signs to a file for lqr10.py to consume
    out_path = os.path.join(os.path.dirname(__file__), "tuned_K.npz")
    np.savez(out_path,
             K=best[3],
             Q=np.diag(best[1]),
             R=np.diag(best[2]),
             signs=np.array(signs))
    print(f"\nSaved best result to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f} s")
