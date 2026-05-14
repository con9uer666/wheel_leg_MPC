#!/usr/bin/env python3
"""
Wheel-leg LQR simulation entry point.

Usage (with MuJoCo 3D viewer, requires mjpython on macOS):
    mjpython main.py [options]

Usage (headless):
    python main.py --no-view [--no-dashboard] [options]

Options:
    --no-view          Disable MuJoCo 3D viewer
    --no-dashboard     Disable matplotlib real-time dashboard
    --h-ref  FLOAT     Desired body height above ground (default: 0.21 m)
    --vx-ref FLOAT     Initial forward velocity reference (default: 0.0 m/s)
    --duration FLOAT   Simulation duration in seconds (default: 30.0)
"""

import argparse
import os
import sys
import time

# ── macOS: ensure MuJoCo can initialise before importing anything else ──
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path so sub-packages resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from sim.runner import SimRunner
from sim.command import ControlCommand


def parse_args():
    p = argparse.ArgumentParser(description="Wheel-Leg LQR Simulation")
    p.add_argument("--no-view",      action="store_true",
                   help="Disable MuJoCo 3D viewer")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Disable real-time matplotlib dashboard")
    p.add_argument("--h-ref",   type=float, default=0.21,
                   help="Desired body height (m); 0.15 leg + 0.06 wheel")
    p.add_argument("--vx-ref",  type=float, default=0.0,
                   help="Initial velocity reference (m/s)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Simulation duration (s); 0 = run forever")
    p.add_argument("--disturb", type=float, default=0.0, metavar="N",
                   help="Apply alternating ±N forward/backward pushes on the "
                        "trunk every 3 s for 0.2 s (e.g. --disturb 50). 0 = off.")
    p.add_argument("--no-monitor", action="store_true",
                   help="Disable the terminal-side LQR state monitor "
                        "(enabled by default when viewer is up).")
    p.add_argument("--controller", choices=["lqr", "mpc"], default="lqr",
                   help="Controller to use: lqr (default, fixed gain from MATLAB) "
                        "or mpc (online 20-step QP at 100 Hz).")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "models", "wheel_legged.xml")

    if args.controller == "mpc":
        from controllers.mpc10 import MPC10Controller
        print("Initialising MPC controller (CVXPY+OSQP, first solve compiles QP)…")
        t_init = time.time()
        controller = MPC10Controller()
        print(f"MPC ready. N={controller.N}, DT={controller.DT}s "
              f"({1/controller.DT:.0f} Hz),  init={time.time()-t_init:.2f}s")
    else:
        from controllers.lqr10 import LQR10Controller
        print("Initialising LQR controller (pre-computed K matrix from MATLAB)…")
        controller = LQR10Controller()
        print(f"LQR ready. K shape: {controller.K.shape},  "
              f"max|K|={np.abs(controller.K).max():.2f}")

    cmd = ControlCommand(vx_ref=args.vx_ref, h_ref=args.h_ref)
    runner = SimRunner(model_path, controller)

    # ── Optional MuJoCo viewer ─────────────────────────────────────────
    viewer = None
    if not args.no_view:
        try:
            viewer = mujoco.viewer.launch_passive(runner.model, runner.data)
            print("MuJoCo viewer launched.")
        except Exception as e:
            print(f"Viewer unavailable ({e}); running headless.")

    # ── Optional dashboard ──────────────────────────────────────────────
    # macOS matplotlib needs the main thread, but mjpython's viewer also owns
    # the main thread → can't run both. Skip dashboard when viewer is up.
    if viewer is not None and not args.no_dashboard:
        print("Dashboard disabled (incompatible with MuJoCo viewer on macOS).")
        args.no_dashboard = True
    if not args.no_dashboard:
        try:
            from viz.dashboard import RealtimeDashboard
            dash = RealtimeDashboard(runner.buf, cmd)
            dash.start_thread()
            print("Dashboard started.")
        except Exception as e:
            print(f"Dashboard unavailable ({e}); continuing without it.")
            args.no_dashboard = True

    # ── Simulation loop ────────────────────────────────────────────────
    dur_str = "∞" if args.duration <= 0 else f"{args.duration} s"
    print(f"Simulating for {dur_str} …  (Ctrl+C to stop)")
    disturb = None
    if args.disturb > 0:
        disturb = {"force_n": args.disturb, "t_start": 2.0,
                   "duration": 0.2, "period": 3.0}
        print(f"Disturbance enabled: ±{args.disturb:.0f} N for 0.2 s every 3 s "
              f"(starting at t=2 s).")

    # ── Terminal LQR-state monitor ───────────────────────────────────────
    monitor = None
    if viewer is not None and not args.no_monitor:
        try:
            from viz.state_monitor import TerminalStateMonitor
            from sim.state import get_joint_ids
            monitor = TerminalStateMonitor(
                runner.model, runner.data, get_joint_ids(runner.model), cmd)
            monitor.start()
        except Exception as e:
            print(f"State monitor unavailable ({e}).")

    t_wall_start = time.time()
    try:
        runner.run(args.duration, cmd, viewer=viewer, disturb=disturb)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if monitor is not None:
            monitor.stop()
    elapsed = time.time() - t_wall_start
    if args.duration > 0:
        print(f"Done. Wall time: {elapsed:.1f} s  "
              f"(speed: {args.duration / elapsed:.1f}× real-time)")
    else:
        print(f"Stopped after {elapsed:.1f} s wall time.")

    if viewer is not None:
        input("Simulation complete. Press Enter to close viewer…")
        viewer.close()


if __name__ == "__main__":
    main()
