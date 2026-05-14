"""
Load tuning parameters from config.yaml.

Loaded once at simulation startup (no hot-reload). Restart the simulation
after editing config.yaml.

Usage
-----
    from config import CFG
    print(CFG.mpc.N)         # 20
    print(CFG.signs.S)       # +1
    print(CFG.mpc.Q)         # np.ndarray (10,)

The module-level CFG is created at import time so controllers can do
`from config import CFG` and read values during their __init__.
"""

import os
import numpy as np

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required to load config.yaml. "
        "Install with `pip install pyyaml`."
    ) from e


_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class _Section:
    """Dot-access wrapper around a dict. Arrays are converted to numpy."""

    def __init__(self, data: dict):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _Section(v))
            elif isinstance(v, list):
                setattr(self, k, np.asarray(v, dtype=float))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return f"_Section({self.__dict__})"


class Config:
    """Top-level configuration container."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = os.path.abspath(path)
        with open(self.path, "r") as f:
            raw = yaml.safe_load(f)

        for section_name, section_data in raw.items():
            setattr(self, section_name, _Section(section_data))

    def summary(self) -> str:
        """Compact one-screen summary of the most important tuning values."""
        m, l, p, a, o, s = (self.mpc, self.lqr, self.leg_pd,
                            self.actuator, self.osqp, self.signs)
        return "\n".join([
            f"Config loaded from {self.path}",
            f"  MPC:      N={m.N}  DT={m.DT}s ({1/m.DT:.0f} Hz)  LEG_LEN={m.LEG_LEN}m",
            f"  MPC Q:    {m.Q.tolist()}",
            f"  MPC R:    {m.R.tolist()}",
            f"  LegPD:    K_P={p.K_P}  K_D={p.K_D}  FF={p.FF}",
            f"  U_MAX:    {a.U_MAX.tolist()}",
            f"  OSQP:     max_iter={o.max_iter}  eps={o.eps_abs:.0e}  rho={o.rho}  warm={o.warm_start}",
            f"  Signs:    S={int(s.S):+d} PHI={int(s.PHI):+d} TLL={int(s.THETA_LL):+d}"
            f" TB={int(s.THETA_B):+d} HIP_OUT={int(s.HIP_OUT):+d} WHL_OUT={int(s.WHL_OUT):+d}",
        ])


# Module-level singleton. Loaded once at first import.
# Set the CONFIG_PATH env var or pass --config to override (handled in main.py).
_path = os.environ.get("WHEEL_LEG_CONFIG", _DEFAULT_PATH)
CFG = Config(_path)


if __name__ == "__main__":
    print(CFG.summary())
