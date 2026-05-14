"""MuJoCo simulation runner with real-time data buffering."""

import time
from collections import deque

import mujoco
import numpy as np

BUFFER_LEN = 10000   # 20 s of history at 500 Hz control rate


def _make_buffer() -> dict:
    keys = [
        "t",
        "theta_b", "dtheta_b",
        "phi", "dphi",
        "theta_ll", "theta_lr",
        "ds", "vx_ref",
        "L_l", "L_r", "L_ref_l", "L_ref_r",
        "T_wl", "T_wr", "T_bl", "T_br",
    ]
    return {k: deque(maxlen=BUFFER_LEN) for k in keys}


class SimRunner:
    DT_SIM  = 0.002   # physics timestep (s) – must match XML
    DT_CTRL = 0.002   # control period (s) – 500 Hz (= physics rate)

    def __init__(self, model_path: str, controller):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.controller = controller
        self.buf = _make_buffer()

        # Initialise robot upright: trunk z = wheel_radius + leg_length
        # R_w=0.06, leg=0.15 → 0.21 m; small margin so the wheel starts off-floor
        self.data.qpos[2] = 0.215

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.215
        for v in self.buf.values():
            v.clear()

    def run(self, duration: float, cmd, viewer=None, disturb=None):
        """
        Run simulation for `duration` seconds. duration <= 0 runs forever
        (Ctrl+C to stop); used by `--duration 0` for live viewer sessions.

        cmd:      ControlCommand (read by controller each control step)
        viewer:   mujoco passive viewer or None
        disturb:  optional dict {force_n, t_start, duration, period} that
                  periodically pushes the trunk in +x to test robustness.
        """
        trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        t = 0.0
        t_last_ctrl = -self.DT_CTRL  # trigger control on first step
        ctrl = np.zeros(6)
        L_ref_l = 0.15
        L_ref_r = 0.15
        t_wall_start = time.time()
        realtime = viewer is not None
        infinite = duration <= 0

        while infinite or t < duration:
            if t - t_last_ctrl >= self.DT_CTRL - 1e-9:
                ctrl = self.controller.compute(self.model, self.data, cmd)
                t_last_ctrl = t

                # Log to buffer (GIL-safe deque appends)
                from sim.state import extract_state, get_joint_ids
                if not hasattr(self, "_jids"):
                    self._jids = get_joint_ids(self.model)
                st = extract_state(self.model, self.data, self._jids)

                u = self.controller.last_u   # [T_wl, T_wr, T_bl, T_br]

                from controllers.params import R_W, L_MIN, L_MAX
                cos_l = max(np.cos(st["theta_ll"]), 0.1)
                cos_r = max(np.cos(st["theta_lr"]), 0.1)
                L_ref_l = float(np.clip((cmd.h_ref - R_W) / cos_l, L_MIN, L_MAX))
                L_ref_r = float(np.clip((cmd.h_ref - R_W) / cos_r, L_MIN, L_MAX))

                self.buf["t"].append(t)
                self.buf["theta_b"].append(np.degrees(st["theta_b"]))
                self.buf["dtheta_b"].append(st["dtheta_b"])
                self.buf["phi"].append(np.degrees(st["phi"]))
                self.buf["dphi"].append(st["dphi"])
                self.buf["theta_ll"].append(np.degrees(st["theta_ll"]))
                self.buf["theta_lr"].append(np.degrees(st["theta_lr"]))
                self.buf["ds"].append(st["ds"])
                self.buf["vx_ref"].append(cmd.vx_ref)
                self.buf["L_l"].append(st["L_l"])
                self.buf["L_r"].append(st["L_r"])
                # MPC14 setpoints — older MPC10/LQR uses h_ref-driven L_ref
                # (computed above as L_ref_l/L_ref_r) but MPC14 takes L_ref
                # directly from cmd.  Log both so the UI shows whichever is
                # active. cmd.L_ref_* is what MPC14 actually tracks.
                self.buf["L_ref_l"].append(getattr(cmd, "L_ref_l", L_ref_l))
                self.buf["L_ref_r"].append(getattr(cmd, "L_ref_r", L_ref_r))
                self.buf["T_wl"].append(float(u[0]))
                self.buf["T_wr"].append(float(u[1]))
                self.buf["T_bl"].append(float(u[2]))
                self.buf["T_br"].append(float(u[3]))

            # Periodic disturbance push on the trunk
            if disturb is not None:
                t_phase = (t - disturb["t_start"]) % disturb["period"]
                if 0 <= t_phase < disturb["duration"] and t >= disturb["t_start"]:
                    direction = +1 if int((t - disturb["t_start"]) // disturb["period"]) % 2 == 0 else -1
                    self.data.xfrc_applied[trunk_id] = [direction * disturb["force_n"], 0, 0, 0, 0, 0]
                else:
                    self.data.xfrc_applied[trunk_id] = [0, 0, 0, 0, 0, 0]

            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            t += self.DT_SIM

            # Viewer sync at ~60 Hz (not every 2 ms — would starve sim)
            if viewer is not None and (t - getattr(self, "_t_last_sync", 0.0)) >= 1.0 / 60.0:
                viewer.sync()
                self._t_last_sync = t

            # Real-time pacing when viewer is attached
            if realtime:
                wall_elapsed = time.time() - t_wall_start
                sleep_dt = t - wall_elapsed
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
