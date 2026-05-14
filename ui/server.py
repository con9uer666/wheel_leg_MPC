"""
FastAPI + WebSocket UI server for setting MPC references in real time.

Architecture:

    Browser  ←─ WS /ws ─→  uvicorn  ──→  ControlCommand  ──→  MPC
       │                                                       │
       └──────── WS /ws ←─── 30 Hz status push ←─── SimRunner.buf

The simulation main thread runs MuJoCo + MPC at 500 Hz; this server runs
in a background daemon thread and communicates with the simulation via:
  * a shared ControlCommand dataclass (browser → sim)
  * SimRunner.buf deques (sim → browser)

Threading is safe because:
  * Python's GIL makes single-field float writes/reads atomic
  * collections.deque.append is GIL-atomic
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from sim.command import ControlCommand


_HTML_PATH = Path(__file__).with_name("index.html")


def _build_app(cmd: ControlCommand, buf: dict, push_rate_hz: float = 30.0) -> FastAPI:
    app = FastAPI()
    connections: Set[WebSocket] = set()

    @app.get("/")
    async def index():
        return HTMLResponse(_HTML_PATH.read_text())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        connections.add(ws)
        # Send current setpoints on connect so the UI sliders sync
        try:
            await ws.send_text(json.dumps({
                "kind": "setpoints",
                "vx_ref":  cmd.vx_ref,
                "yaw_ref": cmd.yaw_ref,
                "h_ref":   cmd.h_ref,
                "L_ref_l": cmd.L_ref_l,
                "L_ref_r": cmd.L_ref_r,
            }))
        except Exception:
            pass

        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                # Inbound message: setpoint update
                # { "vx_ref": 0.3, "yaw_ref": 0.0, ... }  any subset
                if "vx_ref"  in data: cmd.vx_ref  = float(data["vx_ref"])
                if "yaw_ref" in data: cmd.yaw_ref = float(data["yaw_ref"])
                if "h_ref"   in data: cmd.h_ref   = float(data["h_ref"])
                if "L_ref_l" in data: cmd.L_ref_l = float(data["L_ref_l"])
                if "L_ref_r" in data: cmd.L_ref_r = float(data["L_ref_r"])
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            connections.discard(ws)

    # Background: push status to all connected clients at push_rate_hz
    async def pusher():
        period = 1.0 / push_rate_hz
        while True:
            await asyncio.sleep(period)
            if not connections:
                continue
            # Snapshot the latest values from the deque buffer
            try:
                t = buf["t"][-1] if buf["t"] else 0.0
                payload = {
                    "kind": "status",
                    "t":        t,
                    "pitch":    buf["theta_b"][-1] if buf["theta_b"] else 0.0,
                    "vx":       buf["ds"][-1]      if buf["ds"]      else 0.0,
                    "vx_ref":   buf["vx_ref"][-1]  if buf["vx_ref"]  else 0.0,
                    "yaw":      buf["phi"][-1]     if buf["phi"]     else 0.0,
                    "L_l":      buf["L_l"][-1]     if buf["L_l"]     else 0.0,
                    "L_r":      buf["L_r"][-1]     if buf["L_r"]     else 0.0,
                    "L_ref_l":  buf["L_ref_l"][-1] if buf["L_ref_l"] else 0.0,
                    "L_ref_r":  buf["L_ref_r"][-1] if buf["L_ref_r"] else 0.0,
                    "T_wl":     buf["T_wl"][-1]    if buf["T_wl"]    else 0.0,
                    "T_bl":     buf["T_bl"][-1]    if buf["T_bl"]    else 0.0,
                    "hip_L":    buf["theta_ll"][-1] if buf["theta_ll"] else 0.0,
                    "hip_R":    buf["theta_lr"][-1] if buf["theta_lr"] else 0.0,
                    "yaw_ref":  cmd.yaw_ref,
                }
                msg = json.dumps(payload)
            except Exception:
                continue

            dead = []
            for ws in connections:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                connections.discard(ws)

    @app.on_event("startup")
    async def start_pusher():
        asyncio.create_task(pusher())

    return app


def start_server(cmd: ControlCommand, buf: dict,
                 host: str = "127.0.0.1", port: int = 8000,
                 push_rate_hz: float = 30.0):
    """Spawn uvicorn in this thread (call from `threading.Thread(target=...)`)."""
    app = _build_app(cmd, buf, push_rate_hz)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False, ws_max_size=2**20)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
