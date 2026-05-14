"""
Real-time terminal display of the 10 LQR state observables.

Runs in a background daemon thread and uses ANSI escape codes to update
a fixed block of lines in-place at 10 Hz. Pair this with the MuJoCo
viewer by positioning the launching terminal next to the viewer window.
"""

import threading
import time

import numpy as np
import mujoco

from sim.state import extract_state, pitch_from_xmat


# ── ANSI escape sequences ─────────────────────────────────────────────────
CSI          = "\033["
CURSOR_UP    = lambda n: f"{CSI}{n}A"
CLEAR_LINE   = CSI + "2K"
HIDE_CURSOR  = CSI + "?25l"
SHOW_CURSOR  = CSI + "?25h"
RESET        = CSI + "0m"
BOLD         = CSI + "1m"
CYAN         = CSI + "36m"
YELLOW       = CSI + "33m"
GREEN        = CSI + "32m"
GREY         = CSI + "90m"

N_LINES = 16   # total lines reserved for the display block


def _color(value: float, warn: float, crit: float) -> str:
    """Pick a colour based on |value|."""
    v = abs(value)
    if v >= crit:
        return CSI + "31m"   # red
    if v >= warn:
        return YELLOW
    return GREEN


class TerminalStateMonitor:
    """
    Renders the 10-state LQR observation vector to the terminal in real-time.

    State layout matches controllers/lqr10.py exactly:
        x = [s, ds, phi, dphi, theta_ll, dtheta_ll,
             theta_lr, dtheta_lr, theta_b, dtheta_b]
    """

    REFRESH_HZ = 10.0

    def __init__(self, model, data, joint_ids, cmd=None):
        self.model = model
        self.data  = data
        self.jids  = joint_ids
        self.cmd   = cmd        # ControlCommand for showing setpoints
        self._stop = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        # Show cursor again on exit
        print(SHOW_CURSOR, end="", flush=True)

    # ── render loop ──────────────────────────────────────────────────────
    def _loop(self):
        # Reserve space for the display by scrolling
        print("\n" * (N_LINES - 1), flush=True)
        print(HIDE_CURSOR, end="", flush=True)
        try:
            while not self._stop:
                self._render()
                time.sleep(1.0 / self.REFRESH_HZ)
        finally:
            print(SHOW_CURSOR, end="", flush=True)

    def _render(self):
        try:
            st = extract_state(self.model, self.data, self.jids)
        except Exception:
            return

        # Convert to display units
        s    = st["s"]
        ds   = st["ds"]
        phi  = np.degrees(st["phi"]);       dphi = np.degrees(st["dphi"])
        tll  = np.degrees(st["theta_ll"]);  dtll = np.degrees(st["dtheta_ll"])
        tlr  = np.degrees(st["theta_lr"]);  dtlr = np.degrees(st["dtheta_lr"])
        tb   = np.degrees(st["theta_b"]);   dtb  = np.degrees(st["dtheta_b"])
        L_l  = st["L_l"];                   L_r  = st["L_r"]

        # Refs (if cmd attached)
        vx_ref  = self.cmd.vx_ref  if self.cmd else 0.0
        yaw_ref = np.degrees(self.cmd.yaw_ref) if self.cmd else 0.0
        h_ref   = self.cmd.h_ref   if self.cmd else 0.0

        # Build lines
        c_pitch = _color(tb, warn=2, crit=10)
        c_yaw   = _color(phi - yaw_ref, warn=2, crit=10)
        c_vel   = _color(ds - vx_ref,   warn=0.2, crit=0.5)

        lines = [
            f"{BOLD}{CYAN}╭─────────────────── LQR STATE  (10 observables) ───────────────────╮{RESET}",
            f"  {GREY}slot       symbol      value           ref       delta{RESET}",
            f"  {BOLD}x[0]{RESET}  s         (m)  {s:+11.4f}",
            f"  {BOLD}x[1]{RESET}  ds       (m/s) {c_vel}{ds:+11.4f}{RESET}   {vx_ref:+7.3f}   {ds - vx_ref:+7.3f}",
            f"  {BOLD}x[2]{RESET}  phi       (°)  {c_yaw}{phi:+11.3f}{RESET}   {yaw_ref:+7.2f}   {phi - yaw_ref:+7.2f}",
            f"  {BOLD}x[3]{RESET}  dphi    (°/s)  {dphi:+11.3f}",
            f"  {BOLD}x[4]{RESET}  θ_ll      (°)  {tll:+11.3f}",
            f"  {BOLD}x[5]{RESET}  dθ_ll   (°/s)  {dtll:+11.3f}",
            f"  {BOLD}x[6]{RESET}  θ_lr      (°)  {tlr:+11.3f}",
            f"  {BOLD}x[7]{RESET}  dθ_lr   (°/s)  {dtlr:+11.3f}",
            f"  {BOLD}x[8]{RESET}  θ_b pitch (°)  {c_pitch}{tb:+11.3f}{RESET}",
            f"  {BOLD}x[9]{RESET}  dθ_b    (°/s)  {dtb:+11.3f}",
            f"  {GREY}─── Leg PD ────────────────────────────────────────────────────{RESET}",
            f"        L_l  (m)   {L_l:+11.4f}   L_r  (m)  {L_r:+11.4f}   target h={h_ref:.3f}",
            f"{BOLD}{CYAN}╰───────────────────────────────────────────────────────────────────╯{RESET}",
        ]
        # Make sure we have exactly N_LINES-1 content lines (1 line for cursor)
        while len(lines) < N_LINES - 1:
            lines.append("")

        # Move cursor up N_LINES-1 and rewrite the block
        out = CURSOR_UP(N_LINES - 1)
        for ln in lines[:N_LINES - 1]:
            out += CLEAR_LINE + ln + "\n"
        print(out, end="", flush=True)
