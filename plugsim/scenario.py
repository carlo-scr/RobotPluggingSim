"""Per-episode randomisation: nominal domain randomisation and out-of-distribution (OOD) shifts.

A `Scenario` fully determines an episode's initial conditions (together with the EnvCfg), so any
episode can be rebuilt exactly from its annotations.json, which is what makes counterfactual
re-simulation from saved states possible.

Nominal randomisation ("in distribution") covers: target port, cable layout, cable/boot colour from a
nominal palette, port occupancy (0-50 % of the other ports already have colour-coded cables plugged in),
lighting intensity/tint, small rack-pose offsets (the base never parks identically), mild camera
jitter and sensor noise, occasional shelf clutter and a loose second cable.
Each OOD type pushes one factor outside those ranges (see OOD_TYPES / OOD_GROUPS).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .config import EnvCfg

NOMINAL_CABLE_COLORS = ("blue", "grey", "yellow", "green", "white")
OOD_CABLE_COLORS = ("red", "orange", "purple", "pink", "cyan", "black")
OCCUPANT_COLORS = ("blue", "grey", "yellow", "green", "white", "black")
TARGET_LED = (0.15, 0.45, 1.0, 1)
LED_OFF = (0.06, 0.07, 0.08, 1)
BASE_LIGHTS = {"key": 0.6, "fill": 0.25, "ceil_a": 0.2, "ceil_b": 0.2, "spot": 0.0, "back": 0.0}

OOD_GROUPS = {
    "lighting": ("lighting_dim", "lighting_harsh", "lighting_colored", "lighting_flicker"),
    "appearance": ("cable_color", "boot_color", "panel_color", "clutter", "distractor_cable", "dense_ports", "led_distractors"),
    "sensor": ("camera_shift", "camera_noise", "motion_blur", "overexposure", "lens_occlusion"),
    "physics": ("heavy_cable", "stiff_cable", "slippery", "rack_offset", "arm_gain_drift", "weak_gripper"),
}
OOD_TYPES = tuple(t for g in OOD_GROUPS.values() for t in g)
OOD_ALIASES = {"lighting": "lighting_dim", "distractors": "clutter"}  # names used in v0.1


@dataclass
class Scenario:
    seed: int = 0
    port_idx: int = 0
    occupied: list = field(default_factory=list)  # [[port, r, g, b, a], ...] already-plugged cables
    # Cable layout on the shelf (planar): switch position/heading and per-joint turn angles.
    anchor_side: float = 1.0
    anchor_x: float = 0.44
    theta0: float = -1.57
    deltas: list = field(default_factory=list)
    plug_delta: float = 0.0
    # Appearance
    cable_color: tuple = (0.15, 0.35, 0.85, 1)
    boot_color: tuple = (0.15, 0.35, 0.85, 1)
    panel_rgba: tuple = (0.08, 0.08, 0.09, 1)
    mat_tint: float = 1.0
    lights: dict = field(default_factory=lambda: {k: [v] * 3 for k, v in BASE_LIGHTS.items()})
    flicker: list = field(default_factory=list)  # [amplitude, frequency Hz, phase] or []
    headlight: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])  # [scale, r, g, b] of the ambient headlight
    target_led: tuple = TARGET_LED
    led_distractors: list = field(default_factory=list)  # [[port, r, g, b, a], ...]
    clutter: list = field(default_factory=list)  # [[object index, x, y, yaw], ...]
    loose_cable: dict = field(default_factory=dict)  # {"points": [[x, y], ...], "color": rgba}
    # Sensors
    cam_jitter: dict = field(default_factory=dict)  # camera -> [dx,dy,dz, rx,ry,rz]
    image_fx: dict = field(default_factory=dict)  # see env.apply_image_fx
    # Physics / robot
    rack_pose: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # dx, dy, dz, yaw
    pad_friction_scale: float = 1.0
    cable_mass_scale: float = 1.0
    cable_stiffness_scale: float = 1.0
    arm_gain_scale: list = field(default_factory=lambda: [1.0, 1.0])  # (right, left) kp multiplier
    grip_force_scale: float = 1.0
    home_noise: list = field(default_factory=list)  # 14 joint offsets
    ood: list = field(default_factory=list)

    def to_dict(self):
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    @staticmethod
    def from_dict(d):
        known = {f.name for f in dataclasses.fields(Scenario)}
        return Scenario(**{k: v for k, v in d.items() if k in known})


def cable_points(cfg: EnvCfg, anchor_xy, theta0, deltas, plug_delta):
    """Planar forward kinematics of the cable chain. Returns link endpoints, plug rear, plug heading."""
    L = cfg.cable.link_len
    phi = theta0 + np.cumsum(deltas)
    steps = L * np.stack([np.cos(phi), np.sin(phi)], 1)
    pts = np.vstack([anchor_xy, anchor_xy + np.cumsum(steps, 0)])
    return pts, phi[-1] + plug_delta


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sample_layout(cfg: EnvCfg, rng: np.random.Generator, max_tries=2000):
    """Sample a smooth, non-self-intersecting cable layout on the shelf with a reachable plug.

    Random walk on the chain's planar turn angles: smooth random curvature plus a steering term
    that bends the cable away from the shelf edges, followed by rejection checks.
    """
    s, c, pl = cfg.shelf, cfg.cable, cfg.plug
    n, L = c.n_links, c.link_len
    tip_len = pl.boot_len + pl.body_len
    S = n * L
    sv = (np.arange(n) + 0.5) * L
    m = 0.025
    lo = np.array([s.x_min + m, -s.y_half + m])
    hi = np.array([s.x_max - m, s.y_half - m])
    centre = (lo + hi) / 2
    for _ in range(max_tries):
        side = rng.choice([-1.0, 1.0])
        ax = rng.uniform(0.36, 0.46)
        anchor = np.array([ax, side * c.anchor_abs_y])
        theta0 = -side * np.pi / 2 + rng.uniform(-0.7, 0.7)
        kappa = rng.uniform(-3, 3) + rng.uniform(0, 5) * np.sin(np.pi * rng.integers(1, 4) * sv / S + rng.uniform(0, 2 * np.pi))
        deltas = np.zeros(n)
        pos, h = anchor.copy(), theta0
        for i in range(n):
            d = kappa[i] * L if i > 0 else 0.0
            if i > 0:
                ahead = pos + 0.09 * np.array([np.cos(h + d), np.sin(h + d)])
                gap = min(np.min(ahead - lo), np.min(hi - ahead))
                if gap < 0.05:
                    to_c = np.arctan2(*(centre - pos)[::-1])
                    d += np.clip(_wrap(to_c - h), -0.3, 0.3) * np.clip((0.05 - gap) / 0.05, 0, 1)
                d = float(np.clip(d, -0.35, 0.35))
            deltas[i] = d
            h += d
            pos = pos + L * np.array([np.cos(h), np.sin(h)])
        plug_delta = rng.uniform(-0.15, 0.15)
        pts, heading = cable_points(cfg, anchor, theta0, deltas, plug_delta)
        u = np.array([np.cos(heading), np.sin(heading)])
        tip = pts[-1] + tip_len * u
        grasp = tip - pl.grasp_from_tip * u
        allp = np.vstack([pts, tip])
        if not (np.all(allp > lo) and np.all(allp < hi)):
            continue
        if not (0.25 < grasp[0] < 0.43 and abs(grasp[1]) < 0.22):
            continue
        if abs(_wrap(heading)) > 2.3:
            continue
        D = np.linalg.norm(allp[:, None] - allp[None], axis=-1)
        idx = np.arange(len(allp))
        if np.any(D[np.abs(idx[:, None] - idx[None]) >= 3] < 0.016):
            continue
        if np.linalg.norm(tip - anchor) < 0.12:
            continue
        return dict(anchor_side=float(side), anchor_x=float(ax), theta0=float(theta0), deltas=deltas.tolist(), plug_delta=float(plug_delta))
    raise RuntimeError("could not sample a valid cable layout")


def _loose_cable(cfg: EnvCfg, rng, avoid_xy):
    """A smooth random polyline lying on the shelf (visual distractor)."""
    s = cfg.shelf
    n = cfg.loose_cable_links
    for _ in range(200):
        p = np.array([rng.uniform(s.x_min + 0.04, s.x_max - 0.04), rng.uniform(-s.y_half + 0.05, s.y_half - 0.05)])
        h, k0, k1 = rng.uniform(-np.pi, np.pi), rng.uniform(-6, 6), rng.uniform(0, 8)
        pts = [p.copy()]
        for i in range(n):
            h += (k0 + k1 * np.sin(i / n * np.pi * 2)) * 0.03
            p = p + 0.03 * np.array([np.cos(h), np.sin(h)])
            pts.append(p.copy())
        pts = np.array(pts)
        if np.all((pts[:, 0] > s.x_min + 0.01) & (pts[:, 0] < s.x_max - 0.01) & (np.abs(pts[:, 1]) < s.y_half - 0.01)):
            if np.min(np.linalg.norm(pts - avoid_xy, axis=1)) > 0.05:
                return pts.tolist()
    return []


def sample_scenario(cfg: EnvCfg, seed: int, ood=(), port_idx=None) -> Scenario:
    from .scene import CABLE_COLORS, active_ports

    rng = np.random.default_rng(seed)
    ood = [OOD_ALIASES.get(o, o) for o in ood]
    for o in ood:
        if o not in OOD_TYPES:
            raise ValueError(f"unknown OOD type {o!r}; choose from {OOD_TYPES}")
    sc = Scenario(seed=seed, ood=ood)
    act = active_ports(cfg)
    sc.port_idx = int(rng.choice(act)) if port_idx is None else int(port_idx)
    for k, v in sample_layout(cfg, rng).items():
        setattr(sc, k, v)

    # ---- nominal randomisation ----
    col = CABLE_COLORS[rng.choice(NOMINAL_CABLE_COLORS)]
    sc.cable_color = col
    sc.boot_color = col if rng.random() < 0.7 else CABLE_COLORS[rng.choice(["black", "grey", "white"])]
    occ_frac = rng.uniform(0.0, 0.5)
    others = [k for k in range(cfg.rack.n_ports) if k != sc.port_idx]
    sc.occupied = [[int(k), *CABLE_COLORS[rng.choice(OCCUPANT_COLORS)]] for k in others if rng.random() < occ_frac]
    g = rng.uniform(0.06, 0.12)
    sc.panel_rgba = (g, g, g + 0.01, 1)
    sc.mat_tint = float(rng.uniform(0.85, 1.05))
    tint = rng.uniform(0.95, 1.0, 3)
    sc.lights = {k: (v * rng.uniform(0.8, 1.2) * tint).tolist() for k, v in BASE_LIGHTS.items()}
    sc.rack_pose = [float(rng.uniform(-0.005, 0.01)), float(rng.uniform(-0.01, 0.01)), float(rng.uniform(-0.008, 0.008)), float(rng.uniform(-0.02, 0.02))]
    cams = cfg.camera.all_cameras
    sc.cam_jitter = {c: np.concatenate([rng.normal(0, 0.0015, 3), rng.normal(0, 0.012, 3)]).tolist() for c in cams}
    sc.image_fx = {"noise": float(rng.uniform(0, 2.0)), "gain": float(rng.uniform(0.95, 1.05))}
    cable_xy = _cable_xy(cfg, sc)
    if rng.random() < 0.3:
        sc.clutter = [_clutter_item(cfg, rng, 0, cable_xy, 0.10)]  # objects are up to ~15 cm long
    if rng.random() < 0.25:
        sc.loose_cable = {"points": _loose_cable(cfg, rng, _plug_xy(cfg, sc)), "color": list(CABLE_COLORS[rng.choice(NOMINAL_CABLE_COLORS)])}
    sc.pad_friction_scale = float(rng.uniform(0.9, 1.1))
    sc.cable_mass_scale = float(rng.uniform(0.9, 1.1))
    sc.cable_stiffness_scale = float(rng.uniform(0.8, 1.25))
    sc.home_noise = rng.normal(0, 0.02, 14).tolist()

    # ---- OOD shifts (outside the nominal ranges above) ----
    if "lighting_dim" in ood:
        s = rng.uniform(0.25, 0.45)
        sc.lights = {k: (np.asarray(v) * s).tolist() for k, v in sc.lights.items()}
        sc.headlight = [float(s), 1.0, 1.0, 1.0]
    if "lighting_harsh" in ood:
        sc.lights.update(key=[0.1] * 3, fill=[0.03] * 3, ceil_a=[0.05] * 3, ceil_b=[0.05] * 3, spot=[float(rng.uniform(1.2, 1.8))] * 3)
        sc.headlight = [0.25, 1.0, 1.0, 1.0]
    if "lighting_colored" in ood:
        hue = rng.choice([[1.0, 0.25, 0.2], [0.3, 0.45, 1.0], [1.0, 0.7, 0.25], [0.4, 1.0, 0.5]])
        sc.lights = {k: (np.asarray(v) * np.asarray(hue) * 1.2).tolist() for k, v in sc.lights.items()}
        sc.headlight = [1.0, *[float(h) for h in hue]]
    if "lighting_flicker" in ood:
        sc.flicker = [float(rng.uniform(0.4, 0.8)), float(rng.uniform(2.0, 8.0)), float(rng.uniform(0, 2 * np.pi))]
    if "cable_color" in ood:
        sc.cable_color = CABLE_COLORS[rng.choice(OOD_CABLE_COLORS)]
        sc.boot_color = sc.cable_color
    if "boot_color" in ood:
        sc.boot_color = tuple(float(x) for x in rng.choice([[1.0, 0.1, 0.8, 1], [0.6, 1.0, 0.1, 1], [1.0, 0.55, 0.0, 1]]))
    if "panel_color" in ood:
        sc.panel_rgba = tuple(float(x) for x in rng.choice([[0.85, 0.85, 0.86, 1], [0.15, 0.25, 0.6, 1], [0.6, 0.6, 0.55, 1]]))
    if "clutter" in ood:
        sc.clutter = [_clutter_item(cfg, rng, i, cable_xy, 0.03) for i in range(int(rng.integers(3, cfg.n_clutter + 1)))]
    if "distractor_cable" in ood:
        sc.loose_cable = {"points": _loose_cable(cfg, rng, _plug_xy(cfg, sc)), "color": list(sc.cable_color)}
    if "dense_ports" in ood:
        frac = rng.uniform(0.85, 1.0)
        sc.occupied = [[int(k), *CABLE_COLORS[rng.choice(OCCUPANT_COLORS)]] for k in others if rng.random() < frac]
    if "led_distractors" in ood:
        free = [k for k in others if k not in {o[0] for o in sc.occupied}]
        pick = rng.choice(free, size=min(len(free), int(rng.integers(3, 9))), replace=False) if free else []
        sc.led_distractors = [[int(k), *TARGET_LED] for k in pick]
    if "camera_shift" in ood:
        for c in cams:
            if "gripper" in c:
                dp, dr = rng.normal(0, 1, 3), rng.normal(0, 1, 3)
                sc.cam_jitter[c] = np.concatenate([dp / np.linalg.norm(dp) * rng.uniform(0.008, 0.015), dr / np.linalg.norm(dr) * rng.uniform(0.10, 0.18)]).tolist()
    if "camera_noise" in ood:
        sc.image_fx.update(noise=float(rng.uniform(8, 20)), salt=float(rng.uniform(0.002, 0.01)))
    if "motion_blur" in ood:
        sc.image_fx.update(blur=int(rng.integers(5, 12)), blur_angle=float(rng.uniform(0, np.pi)))
    if "overexposure" in ood:
        sc.image_fx.update(gain=float(rng.uniform(1.6, 2.4)), gamma=float(rng.uniform(0.6, 0.8)))
    if "lens_occlusion" in ood:
        sc.image_fx.update(occlusion=[float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.25, 0.4))])
    if "heavy_cable" in ood:
        sc.cable_mass_scale = float(rng.uniform(2.5, 4.0))
    if "stiff_cable" in ood:
        sc.cable_stiffness_scale = float(rng.uniform(3.0, 6.0))
    if "slippery" in ood:
        sc.pad_friction_scale = float(rng.uniform(0.2, 0.4))
    if "rack_offset" in ood:
        sgn = rng.choice([-1, 1], 2)
        sc.rack_pose = [float(rng.uniform(0.02, 0.045)), float(sgn[0] * rng.uniform(0.02, 0.04)), float(rng.uniform(-0.015, 0.015)), float(sgn[1] * rng.uniform(0.05, 0.09))]
    if "arm_gain_drift" in ood:
        sc.arm_gain_scale = [float(rng.uniform(0.35, 0.6)), float(rng.uniform(0.35, 0.6))]
    if "weak_gripper" in ood:
        sc.grip_force_scale = float(rng.uniform(0.2, 0.35))
    return sc


def _cable_xy(cfg: EnvCfg, sc: Scenario):
    anchor = np.array([sc.anchor_x, sc.anchor_side * cfg.cable.anchor_abs_y])
    pts, h = cable_points(cfg, anchor, sc.theta0, np.asarray(sc.deltas), sc.plug_delta)
    tip = pts[-1] + (cfg.plug.boot_len + cfg.plug.body_len) * np.array([np.cos(h), np.sin(h)])
    return np.vstack([pts, tip])


def _plug_xy(cfg: EnvCfg, sc: Scenario):
    return _cable_xy(cfg, sc)[-2]


def _clutter_item(cfg: EnvCfg, rng, i, cable_xy, min_d):
    """Clutter position on the shelf, at least `min_d` away from the cable (and the plug)."""
    s = cfg.shelf
    for _ in range(200):
        xy = np.array([rng.uniform(s.x_min + 0.05, s.x_max - 0.05), rng.uniform(-s.y_half + 0.06, s.y_half - 0.06)])
        if np.min(np.linalg.norm(cable_xy - xy, axis=1)) > min_d + 0.04:
            break
    return [int(i), float(xy[0]), float(xy[1]), float(rng.uniform(-np.pi, np.pi))]
