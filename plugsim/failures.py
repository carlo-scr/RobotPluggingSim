"""Known-failure injection specs. The scripted expert reads these and corrupts its own behaviour.

`severity` in [0, 1] scales each perturbation. The *injected* failure and the *physical outcome* are
labelled separately: a mild misalignment can still succeed thanks to the chamfer, and such episodes
stay labelled success.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

FAILURE_TYPES = (
    "drop",  # gripper briefly opens during transport -> cable dropped (the real dataset's main failure)
    "weak_grip",  # low grasp force -> plug slips / rotates in hand under the cable's pull
    "misalign",  # biased port estimate (lateral offset + angle error) -> jam on the face/chamfer
    "partial_insert",  # stops short of the latch depth and lets go -> not latched
    "wrong_port",  # plugs into a neighbouring port
    "misgrasp",  # grasp point offset (onto the plug body, or onto the cable)
    "bump",  # external knock on the gripper during transport/alignment/insertion
)


@dataclass
class FailureSpec:
    kind: str
    severity: float = 0.5
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)


def sample_failure(kind: str, severity: float, rng: np.random.Generator, cfg, scenario) -> FailureSpec:
    """Draws the concrete perturbation parameters for a failure type at a given severity."""
    s = float(np.clip(severity, 0, 1))
    p = {}
    if kind == "drop":
        p["at_frac"] = float(rng.uniform(0.15, 0.85))  # fraction of the transport phase
        p["open_s"] = 0.25 + 0.35 * s
    elif kind == "weak_grip":
        # Fraction of the nominal grip force (ctrl 1.0 = full force, 0.5 = zero force).
        p["force_frac"] = 0.15 * (1 - s) ** 1.5 + 0.01
        p["grip_cmd"] = 0.5 + 0.5 * p["force_frac"]
    elif kind == "misalign":
        ang = rng.uniform(0, 2 * np.pi)
        r = 0.001 + 0.004 * s
        p["dyz"] = [float(r * np.cos(ang)), float(r * np.sin(ang))]
        rv = rng.normal(0, 1, 3)
        rv[0] = 0.0  # roll about the insertion axis matters less
        p["drot"] = (rv / (np.linalg.norm(rv) + 1e-9) * (0.03 + 0.15 * s)).tolist()
    elif kind == "partial_insert":
        p["depth_frac"] = float(0.85 - 0.6 * s)  # fraction of the seat depth reached
    elif kind == "wrong_port":
        # a free physical port next to the target (same row first, then the row above/below)
        from .scene import active_ports, port_yz

        occ = {int(e[0]) for e in scenario.occupied}
        free = [k for k in active_ports(cfg) if k not in occ and k != scenario.port_idx]
        y0, z0 = port_yz(cfg, scenario.port_idx)
        dist = [abs(port_yz(cfg, k)[0] - y0) + 2.0 * abs(port_yz(cfg, k)[1] - z0) for k in free]
        near = [k for k, dd in sorted(zip(free, dist), key=lambda t: t[1])][:2]
        p["port"] = int(rng.choice(near))
    elif kind == "misgrasp":
        mode = str(rng.choice(["forward", "backward"]))
        p["mode"] = mode
        p["offset"] = (0.006 + 0.010 * s) if mode == "forward" else -(0.012 + 0.020 * s)
    elif kind == "bump":
        d = rng.normal(0, 1, 3)
        d[0] = 0.0
        # A sharp knock on the gripper holding the plug (e.g. hitting the rack or the other arm).
        p["force"] = (d / (np.linalg.norm(d) + 1e-9) * (20.0 + 180.0 * s)).tolist()
        tq = rng.normal(0, 1, 3)
        p["torque"] = (tq / (np.linalg.norm(tq) + 1e-9) * (1.0 + 10.0 * s)).tolist()
        p["duration_s"] = 0.05 + 0.10 * s
        p["phase"] = str(rng.choice(["transport", "align", "insert"]))
        p["at_frac"] = float(rng.uniform(0.2, 0.8))
    else:
        raise ValueError(f"unknown failure {kind!r}; choose from {FAILURE_TYPES}")
    return FailureSpec(kind, s, p)
