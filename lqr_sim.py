#!/usr/bin/env python3
"""
LQR balance + velocity controller for wheel-leg robot.
Mirrors main.py (MPC) but uses a closed-form LQR gain instead of online MPC.

Based on MATLAB HerKules_VOCAL_SJ_LQR_v4_with_data.m

Usage (3D viewer, requires mjpython on macOS):
    mjpython lqr_sim.py [options]

Usage (headless):
    python lqr_sim.py --no-view [--no-dashboard] [options]

Options:
    --no-view          Disable MuJoCo 3D viewer
    --no-dashboard     Disable matplotlib real-time dashboard
    --h-ref  FLOAT     Desired body height (default: 0.27 m)
    --vx-ref FLOAT     Initial velocity reference (default: 0.0 m/s)
    --duration FLOAT   Simulation duration (default: 30.0 s)
    --leg    FLOAT     Fixed leg pole length for K computation (default: 0.18 m)
"""

import argparse
import os
import sys
import time
import threading
from collections import deque

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from controllers.model10 import get_AB10
from controllers.params import R_W, L_MIN, L_MAX
from sim.command import ControlCommand


# ── LQR weights (from MATLAB lqr_Q / lqr_R) ─────────────────────────
LQR_Q = np.diag([1, 2, 12000, 200, 1000, 1, 1000, 1, 20000, 1]).astype(float)
LQR_R = np.diag([0.25, 0.25, 1.5, 1.5]).astype(float)

# ── Sign conventions (brute-force verified: pitch<0.09°, vx err<0.01 m/s) ─
SIGN_THETA_LL = -1   # LQR sees -hip_angle
SIGN_THETA_B  = -1   # LQR sees -pitch
SIGN_DS       = -1   # LQR sees -qvel[0]
SIGN_HIP_OUT  = +1   # ctrl_hip = +T_bl
SIGN_WHL_OUT  = -1   # ctrl_wheel = -T_wl

# Actuator torque/force limits matching the XML
U_MAX = np.array([30.0, 30.0, 15.0, 15.0])  # [T_wl, T_wr, T_bl, T_br] (N·m)

# Leg PD gains — ground contact handles gravity passively (no feedforward needed)
K_LEG_P = 2500.0
K_LEG_D = 120.0

BUFFER_LEN = 4000   # ~80 s at 50 Hz control rate
DT_SIM     = 0.002  # physics timestep (must match XML)
DT_CTRL    = 0.02   # LQR control period (50 Hz)


# ─────────────────────────────────────────────────────────────────────
# LQR gain
# ─────────────────────────────────────────────────────────────────────

def compute_lqr_gain(L_l: float = 0.18, L_r: float = 0.18) -> np.ndarray:
    """Solve continuous-time ARE and return 4×10 gain matrix K."""
    A, B = get_AB10(L_l, L_r)
    P = scipy.linalg.solve_continuous_are(A, B, LQR_Q, LQR_R)
    return np.linalg.inv(LQR_R) @ B.T @ P


# ─────────────────────────────────────────────────────────────────────
# State extraction
# ─────────────────────────────────────────────────────────────────────

def _get_jids(model) -> dict:
    out = {}
    for name in ("hip_L", "hip_R", "leg_L", "leg_R", "wheel_L", "wheel_R"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        out[name] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])
    return out


def _pitch_from_xmat(xmat) -> float:
    """Full ±180° pitch from MuJoCo body rotation matrix."""
    return float(np.arctan2(xmat[2], xmat[8]))


def _yaw_from_quat(q) -> float:
    """Yaw from free-joint quaternion [w, x, y, z]."""
    w, x, y, z = q
    return float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def build_lqr_state(model, data, jids: dict) -> np.ndarray:
    """
    10-element state vector with LQR sign conventions:
      [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr, theta_b, dtheta_b]
    """
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    xmat     = data.xmat[trunk_id]

    pitch  = _pitch_from_xmat(xmat)
    dpitch = float(data.qvel[4])
    dyaw   = float(data.qvel[5])
    yaw    = _yaw_from_quat(data.qpos[3:7])

    qa_L, qv_L = jids["hip_L"]
    qa_R, qv_R = jids["hip_R"]
    hip_L = float(data.qpos[qa_L]);  dhip_L = float(data.qvel[qv_L])
    hip_R = float(data.qpos[qa_R]);  dhip_R = float(data.qvel[qv_R])

    s  = float(data.qpos[0])
    ds = float(data.qvel[0]) * SIGN_DS

    return np.array([
        s,                        ds,
        yaw,                      dyaw,
        hip_L * SIGN_THETA_LL,    dhip_L * SIGN_THETA_LL,
        hip_R * SIGN_THETA_LL,    dhip_R * SIGN_THETA_LL,
        pitch * SIGN_THETA_B,     dpitch,
    ])


def build_display_state(model, data, jids: dict) -> dict:
    """Physical state dict for the dashboard buffer (degrees, raw signs)."""
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    xmat  = data.xmat[trunk_id]
    pitch = _pitch_from_xmat(xmat)
    yaw   = _yaw_from_quat(data.qpos[3:7])

    qa_L, qv_L = jids["hip_L"]
    qa_R, qv_R = jids["hip_R"]
    ql_L, dql_L = jids["leg_L"]
    ql_R, dql_R = jids["leg_R"]

    ext_L = float(data.qpos[ql_L]);  dext_L = float(data.qvel[dql_L])
    ext_R = float(data.qpos[ql_R]);  dext_R = float(data.qvel[dql_R])

    return {
        "s":        float(data.qpos[0]),
        "ds":       -float(data.qvel[0]),   # forward positive
        "phi":      np.degrees(yaw),
        "dphi":     float(data.qvel[5]),
        "theta_ll": np.degrees(float(data.qpos[qa_L])),
        "theta_lr": np.degrees(float(data.qpos[qa_R])),
        "theta_b":  np.degrees(pitch),
        "ext_L":    ext_L,  "dext_L": dext_L,
        "ext_R":    ext_R,  "dext_R": dext_R,
        "L_l":      L_MIN + ext_L,
        "L_r":      L_MIN + ext_R,
    }


# ─────────────────────────────────────────────────────────────────────
# Buffer
# ─────────────────────────────────────────────────────────────────────

def _make_buffer() -> dict:
    keys = [
        "t", "theta_b", "dtheta_b", "phi", "dphi",
        "theta_ll", "theta_lr", "ds", "vx_ref",
        "L_l", "L_r", "L_ref_l", "L_ref_r",
        "T_wl", "T_wr", "T_bl", "T_br",
    ]
    return {k: deque(maxlen=BUFFER_LEN) for k in keys}


# ─────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────

def run_sim(model, data, jids, K, cmd, buf, duration, viewer=None) -> float:
    """
    Run simulation for `duration` seconds.
    Returns wall-clock time elapsed.

    Real-time pacing keeps sim time aligned with wall time when a viewer is
    attached, so visualization stays smooth and control loop runs at a
    deterministic 50 Hz instead of getting starved by GIL contention.
    """
    t            = 0.0
    t_last_ctrl  = -DT_CTRL
    t_last_sync  = 0.0
    ctrl         = np.zeros(6)
    infinite     = (duration <= 0)
    realtime     = (viewer is not None)
    SYNC_DT      = 1.0 / 60.0          # ~60 Hz viewer refresh

    print(f"Simulating {'∞' if infinite else f'{duration:.0f}'} s at {1/DT_CTRL:.0f} Hz LQR …")
    t_wall_start = time.time()

    while infinite or t < duration:
        # ── LQR control at DT_CTRL rate ──────────────────────────────
        if t - t_last_ctrl >= DT_CTRL - 1e-9:
            t_last_ctrl = t

            st = build_display_state(model, data, jids)
            x  = build_lqr_state(model, data, jids)

            # Reference: track desired velocity and yaw; balance at 0 otherwise
            x_ref    = np.zeros(10)
            x_ref[1] = cmd.vx_ref    # ds reference
            x_ref[2] = cmd.yaw_ref   # phi reference (rad)

            # LQR: u = -K (x - x_ref)
            u = -K @ (x - x_ref)
            u = np.clip(u, -U_MAX, U_MAX)
            T_wl, T_wr, T_bl, T_br = u

            # Leg length reference from height command
            cos_l   = max(np.cos(np.radians(st["theta_ll"])), 0.1)
            cos_r   = max(np.cos(np.radians(st["theta_lr"])), 0.1)
            L_ref_l = float(np.clip((cmd.h_ref - R_W) / cos_l, L_MIN, L_MAX))
            L_ref_r = float(np.clip((cmd.h_ref - R_W) / cos_r, L_MIN, L_MAX))

            # Leg PD (ground contact handles gravity; no feedforward needed)
            dev_L = (L_ref_l - L_MIN) - st["ext_L"]
            dev_R = (L_ref_r - L_MIN) - st["ext_R"]
            F_l = K_LEG_P * dev_L - K_LEG_D * st["dext_L"]
            F_r = K_LEG_P * dev_R - K_LEG_D * st["dext_R"]

            # MuJoCo ctrl: [hip_L, hip_R, leg_L, leg_R, wheel_L, wheel_R]
            ctrl[0] = T_bl * SIGN_HIP_OUT
            ctrl[1] = T_br * SIGN_HIP_OUT
            ctrl[2] = F_l
            ctrl[3] = F_r
            ctrl[4] = T_wl * SIGN_WHL_OUT
            ctrl[5] = T_wr * SIGN_WHL_OUT

            # Buffer for dashboard
            buf["t"].append(t)
            buf["theta_b"].append(st["theta_b"])
            buf["dtheta_b"].append(float(data.qvel[4]))
            buf["phi"].append(st["phi"])
            buf["dphi"].append(st["dphi"])
            buf["theta_ll"].append(st["theta_ll"])
            buf["theta_lr"].append(st["theta_lr"])
            buf["ds"].append(st["ds"])
            buf["vx_ref"].append(cmd.vx_ref)
            buf["L_l"].append(st["L_l"])
            buf["L_r"].append(st["L_r"])
            buf["L_ref_l"].append(L_ref_l)
            buf["L_ref_r"].append(L_ref_r)
            buf["T_wl"].append(float(T_wl))
            buf["T_wr"].append(float(T_wr))
            buf["T_bl"].append(float(T_bl))
            buf["T_br"].append(float(T_br))

        # ── Physics step ─────────────────────────────────────────────
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        t += DT_SIM

        # ── Viewer sync at ~60 Hz (not every 2ms — would starve sim) ─
        if viewer is not None and (t - t_last_sync) >= SYNC_DT:
            viewer.sync()
            t_last_sync = t

        # ── Real-time pacing when viewer attached ────────────────────
        if realtime:
            wall_elapsed = time.time() - t_wall_start
            sleep_dt = t - wall_elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    return time.time() - t_wall_start


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wheel-Leg LQR Simulation")
    p.add_argument("--no-view",      action="store_true")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--h-ref",    type=float, default=0.27,
                   help="Desired body height (m)")
    p.add_argument("--vx-ref",   type=float, default=0.0,
                   help="Initial velocity reference (m/s)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Simulation duration (s), 0 = infinite")
    p.add_argument("--leg",      type=float, default=0.18,
                   help="Leg length for LQR gain (m)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "models", "wheel_legged.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    data.qpos[2] = 0.285           # start upright
    mujoco.mj_forward(model, data)

    jids = _get_jids(model)

    print(f"Computing LQR gain for L={args.leg:.3f} m …")
    K = compute_lqr_gain(args.leg, args.leg)
    print(f"Done. K shape: {K.shape},  max|K|={np.abs(K).max():.2f}")

    cmd = ControlCommand(vx_ref=args.vx_ref, h_ref=args.h_ref)
    buf = _make_buffer()

    # ── Optional MuJoCo 3D viewer ────────────────────────────────────
    viewer = None
    if not args.no_view:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print("MuJoCo viewer launched.")
        except Exception as e:
            print(f"Viewer unavailable ({e}); running headless.")

    # ── Run: sim in background thread, dashboard on main thread ─────
    result = {}

    def _sim_thread():
        result["elapsed"] = run_sim(
            model, data, jids, K, cmd, buf, args.duration, viewer
        )

    sim_t = threading.Thread(target=_sim_thread, daemon=True)
    sim_t.start()

    if not args.no_dashboard:
        # Dashboard must run on the main thread (macOS GUI requirement)
        from viz.dashboard import RealtimeDashboard
        dash = RealtimeDashboard(buf, cmd)
        dash._init_figure()
        while sim_t.is_alive():
            try:
                dash._update()
            except Exception:
                pass
            try:
                plt.pause(dash.REFRESH_DT)
            except Exception:
                break
        dash.stop()
        plt.close("all")
    else:
        sim_t.join()

    elapsed = result.get("elapsed", 1.0)
    print(f"Done. Wall: {elapsed:.1f}s  ({args.duration/elapsed:.1f}× real-time)")

    trunk_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    pitch_deg = np.degrees(_pitch_from_xmat(data.xmat[trunk_id]))
    print(f"Final pitch: {pitch_deg:+.2f}°   "
          f"vx: {-data.qvel[0]:+.3f} m/s   "
          f"x: {data.qpos[0]:+.3f} m")

    if viewer is not None:
        input("Press Enter to close viewer …")
        viewer.close()


if __name__ == "__main__":
    main()
