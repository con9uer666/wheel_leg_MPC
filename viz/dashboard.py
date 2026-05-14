"""
Real-time matplotlib dashboard with 6 subplots and interactive sliders.

Run in a background daemon thread via start_thread().
Reads data from SimRunner.buf (collections.deque – GIL-safe).
Writes control commands back to a ControlCommand object via slider callbacks.
"""

import threading
import time
import numpy as np

import matplotlib
matplotlib.use("MacOSX")     # Metal-compatible on M1; falls back to TkAgg if needed
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg


class RealtimeDashboard:
    REFRESH_DT = 0.10   # seconds between plot updates
    WINDOW_S   = 8.0    # seconds of history shown

    def __init__(self, buf: dict, cmd):
        """
        buf:  SimRunner.buf  (dict of deques)
        cmd:  ControlCommand (slider callbacks write here)
        """
        self.buf = buf
        self.cmd = cmd
        self._stop = False

    # ── Figure construction ──────────────────────────────────────────────
    def _init_figure(self):
        plt.ion()
        self.fig = plt.figure("Wheel-Leg MPC Dashboard", figsize=(14, 9))
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95,
                                  bottom=0.30, hspace=0.45, wspace=0.35)

        axes = self.fig.subplots(2, 3)
        ax_pitch, ax_vel, ax_legs   = axes[0]
        ax_torq,  ax_len, ax_yaw    = axes[1]

        # ── subplot 0: body pitch ──
        ax_pitch.set_title("Body Pitch θ_b (°)")
        ax_pitch.set_xlabel("t (s)"); ax_pitch.set_ylabel("deg")
        ax_pitch.axhline(0, color="k", lw=0.5)
        self.ln_pitch, = ax_pitch.plot([], [], "b", lw=1.2, label="θ_b")
        ax_pitch.legend(fontsize=7)

        # ── subplot 1: velocity ──
        ax_vel.set_title("Forward Velocity (m/s)")
        ax_vel.set_xlabel("t (s)"); ax_vel.set_ylabel("m/s")
        self.ln_vel, = ax_vel.plot([], [], "b", lw=1.2, label="ṡ")
        self.ln_vref, = ax_vel.plot([], [], "r--", lw=1.0, label="ref")
        ax_vel.legend(fontsize=7)

        # ── subplot 2: leg angles ──
        ax_legs.set_title("Leg Angles (°)")
        ax_legs.set_xlabel("t (s)"); ax_legs.set_ylabel("deg")
        self.ln_thl, = ax_legs.plot([], [], "b", lw=1.2, label="θ_ll")
        self.ln_thr, = ax_legs.plot([], [], "r", lw=1.2, label="θ_lr")
        ax_legs.legend(fontsize=7)

        # ── subplot 3: torques ──
        ax_torq.set_title("Control Torques (N·m)")
        ax_torq.set_xlabel("t (s)"); ax_torq.set_ylabel("N·m")
        self.ln_twl, = ax_torq.plot([], [], lw=1.0, label="T_wl")
        self.ln_twr, = ax_torq.plot([], [], lw=1.0, label="T_wr")
        self.ln_tbl, = ax_torq.plot([], [], lw=1.0, label="T_bl")
        self.ln_tbr, = ax_torq.plot([], [], lw=1.0, label="T_br")
        ax_torq.legend(fontsize=7, ncol=2)

        # ── subplot 4: leg lengths ──
        ax_len.set_title("Leg Lengths (m)")
        ax_len.set_xlabel("t (s)"); ax_len.set_ylabel("m")
        self.ln_ll,  = ax_len.plot([], [], "b", lw=1.2, label="L_l")
        self.ln_lr,  = ax_len.plot([], [], "r", lw=1.2, label="L_r")
        self.ln_lrl, = ax_len.plot([], [], "b--", lw=0.8, label="ref_l")
        self.ln_lrr, = ax_len.plot([], [], "r--", lw=0.8, label="ref_r")
        ax_len.legend(fontsize=7, ncol=2)

        # ── subplot 5: yaw ──
        ax_yaw.set_title("Yaw φ (°)")
        ax_yaw.set_xlabel("t (s)"); ax_yaw.set_ylabel("deg")
        ax_yaw.axhline(0, color="k", lw=0.5)
        self.ln_yaw, = ax_yaw.plot([], [], "g", lw=1.2, label="φ")
        ax_yaw.legend(fontsize=7)

        self._axes = [ax_pitch, ax_vel, ax_legs, ax_torq, ax_len, ax_yaw]

        # ── Interactive sliders ─────────────────────────────────────────
        ax_vx  = self.fig.add_axes([0.10, 0.18, 0.75, 0.03])
        ax_h   = self.fig.add_axes([0.10, 0.12, 0.75, 0.03])
        ax_yaw_sl = self.fig.add_axes([0.10, 0.06, 0.75, 0.03])

        self.sl_vx = wdg.Slider(ax_vx,  "vx_ref (m/s)",  -2.0, 2.0,
                                  valinit=0.0, valstep=0.05)
        self.sl_h  = wdg.Slider(ax_h,   "h_ref  (m)",    0.18, 0.30,
                                  valinit=0.27, valstep=0.005)
        self.sl_yaw= wdg.Slider(ax_yaw_sl, "yaw_ref (°)", -30.0, 30.0,
                                  valinit=0.0, valstep=1.0)

        def on_vx(val):
            self.cmd.vx_ref = float(val)
        def on_h(val):
            self.cmd.h_ref = float(val)
        def on_yaw(val):
            self.cmd.yaw_ref = float(np.deg2rad(val))

        self.sl_vx.on_changed(on_vx)
        self.sl_h.on_changed(on_h)
        self.sl_yaw.on_changed(on_yaw)

    # ── Update logic ────────────────────────────────────────────────────
    def _update(self):
        buf = self.buf
        if len(buf["t"]) < 2:
            return

        t   = np.array(buf["t"])
        t0  = t[-1] - self.WINDOW_S
        mask = t >= t0
        t_w  = t[mask]

        def w(key):
            return np.array(buf[key])[mask]

        def _refresh_line(ln, t_w, y_w):
            ln.set_data(t_w, y_w)
            ln.axes.relim()
            ln.axes.autoscale_view()

        _refresh_line(self.ln_pitch, t_w, w("theta_b"))
        _refresh_line(self.ln_vel,   t_w, w("ds"))
        _refresh_line(self.ln_vref,  t_w, w("vx_ref"))
        _refresh_line(self.ln_thl,   t_w, w("theta_ll"))
        _refresh_line(self.ln_thr,   t_w, w("theta_lr"))
        _refresh_line(self.ln_twl,   t_w, w("T_wl"))
        _refresh_line(self.ln_twr,   t_w, w("T_wr"))
        _refresh_line(self.ln_tbl,   t_w, w("T_bl"))
        _refresh_line(self.ln_tbr,   t_w, w("T_br"))
        _refresh_line(self.ln_ll,    t_w, w("L_l"))
        _refresh_line(self.ln_lr,    t_w, w("L_r"))
        _refresh_line(self.ln_lrl,   t_w, w("L_ref_l"))
        _refresh_line(self.ln_lrr,   t_w, w("L_ref_r"))
        _refresh_line(self.ln_yaw,   t_w, w("phi"))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _loop(self):
        self._init_figure()
        while not self._stop:
            try:
                self._update()
            except Exception:
                pass
            time.sleep(self.REFRESH_DT)

    def start_thread(self):
        t = threading.Thread(target=self._loop, daemon=True, name="dashboard")
        t.start()
        return t

    def stop(self):
        self._stop = True
