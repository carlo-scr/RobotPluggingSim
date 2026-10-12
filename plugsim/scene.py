"""Procedural MJCF for the cable-plugging scene (data-center aisle).

World frame: robot torso at the origin, +x towards the rack, +z up. The rack is its own static body
(pose randomised per episode). In the rack frame the patch-panel face is the plane x = 0, the
insertion direction is +x, and z = 0 is the centre between the panel rows. Primitives + a few convex
meshes, no plugins, so the model stays compatible with MJX / MuJoCo Warp.

Collision groups (contype / conaffinity):
  world, plug, gripper   1 / 7   collide with everything
  cable                  2 / 1   collides with world/plug/gripper, not with itself or arm links
  arm links              4 / 1   collide with world/plug/gripper, not with each other or the cable
  visual-only            0 / 0
"""
from __future__ import annotations

import numpy as np

from . import textures as T
from .config import EnvCfg
from .geometry import fmt, lookat_xyaxes, mat_to_quat

ARMS = ("right", "left")
ARM_SIGN = {"right": -1.0, "left": 1.0}  # y-side of each arm

# Franka-Panda-like kinematic chain (menagerie convention): (pos, quat) of each link in its parent.
_PANDA_CHAIN = [
    ((0, 0, 0.333), (1, 0, 0, 0)),
    ((0, 0, 0), (1, -1, 0, 0)),
    ((0, -0.316, 0), (1, 1, 0, 0)),
    ((0.0825, 0, 0), (1, 1, 0, 0)),
    ((-0.0825, 0.384, 0), (1, -1, 0, 0)),
    ((0, 0, 0), (1, 1, 0, 0)),
    ((0.088, 0, 0), (1, 1, 0, 0)),
]
_PANDA_RANGE = [
    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
    (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973),
]
_PANDA_FORCE = [87, 87, 87, 87, 12, 12, 12]
# Capsules approximating each link's volume, in the link's own frame: (from, to, radius)
_PANDA_GEOMS = [
    [((0, 0, -0.13), (0, 0, 0.0), 0.060)],
    [((0, 0, 0), (0, -0.12, 0), 0.060)],
    [((0, 0, -0.20), (0, 0, 0), 0.055), ((0, 0, 0), (0.0825, 0, 0), 0.050)],
    [((0, 0, 0), (-0.0825, 0.10, 0), 0.055)],
    [((0, 0, -0.27), (0, 0, -0.02), 0.050)],
    [((0, 0, 0), (0.088, 0, 0), 0.045)],
    [((0, 0, -0.02), (0, 0, 0.08), 0.040)],
]
_PANDA_MASS = [3.5, 1.5, 3.0, 3.0, 2.0, 1.5, 0.7]

# Patch-cable colour palette (data centres colour-code by function).
CABLE_COLORS = {
    "blue": (0.15, 0.35, 0.85, 1),
    "grey": (0.55, 0.57, 0.6, 1),
    "yellow": (0.95, 0.8, 0.15, 1),
    "green": (0.2, 0.65, 0.3, 1),
    "white": (0.92, 0.92, 0.9, 1),
    "black": (0.08, 0.08, 0.09, 1),
    "red": (0.85, 0.12, 0.12, 1),
    "orange": (1.0, 0.45, 0.05, 1),
    "purple": (0.5, 0.2, 0.7, 1),
    "pink": (0.95, 0.45, 0.65, 1),
    "cyan": (0.1, 0.75, 0.85, 1),
}
FRONT_QUAT = mat_to_quat(T.FRONT_MAT)
LIGHTS = ("key", "fill", "ceil_a", "ceil_b", "spot", "back")
# Occupant cable stub (droop from the boot rear), in the port frame.
_STUB = np.array([[-0.022, 0, 0], [-0.040, 0, -0.004], [-0.054, 0, -0.024], [-0.061, 0, -0.068], [-0.063, 0, -0.17]])


# ---------------------------------------------------------------------- panel layout helpers
def port_rc(cfg: EnvCfg, k: int):
    return divmod(k, cfg.rack.cols)


def port_yz(cfg: EnvCfg, k: int):
    """(y, z) of port k in the rack frame. Row 0 = top row, column 0 = robot's left (+y)."""
    r, c = port_rc(cfg, k)
    rk = cfg.rack
    return ((rk.cols - 1) / 2 - c) * rk.port_pitch, ((rk.rows - 1) / 2 - r) * rk.row_pitch


def port_y(cfg: EnvCfg, k: int) -> float:
    return port_yz(cfg, k)[0]


def active_ports(cfg: EnvCfg):
    """Physical jacks (possible targets): central columns within active_half_width."""
    return [k for k in range(cfg.rack.n_ports) if abs(port_yz(cfg, k)[0]) <= cfg.rack.active_half_width + 1e-9]


def _box(name, pos, size, cls="world", rgba=None, euler=None, quat=None, extra=""):
    s = f'<geom name="{name}" type="box" class="{cls}" pos="{fmt(pos)}" size="{fmt(size)}"'
    if euler is not None:
        s += f' euler="{fmt(euler)}"'
    if quat is not None:
        s += f' quat="{fmt(quat)}"'
    if rgba is not None:
        s += f' rgba="{fmt(rgba, 3)}"'
    return s + f" {extra}/>"


def _front(name, pos, half_wh, material):
    """Visual textured plane facing the robot (-x normal). half_wh = (half width along y, half height)."""
    # the plane's 3rd size is the *rendering grid spacing*: make it larger than the plane -> one quad
    return (f'<geom name="{name}" type="plane" class="visual" pos="{fmt(pos)}" size="{fmt(half_wh)} {2 * max(half_wh):.3f}" '
            f'quat="{fmt(FRONT_QUAT)}" material="{material}"/>')


# ---------------------------------------------------------------------- rack
def _panel_xml(cfg: EnvCfg):
    """Patch panels, jacks, LEDs, labels, and occupant plugs. Returns (xml, assets)."""
    rk, p, pl = cfg.rack, cfg.port, cfg.plug
    W, H = pl.body_w + p.clearance, pl.body_h + p.clearance
    t, c, w, D = rk.panel_thickness, p.chamfer, p.wall, p.depth
    hw = rk.panel_half_width
    act = set(active_ports(cfg))
    out, assets = [], []
    s2 = np.sqrt(2) / 2
    tip_len = pl.boot_len + pl.body_len
    rng = np.random.default_rng(0)
    for r in range(rk.rows):
        zr = ((rk.rows - 1) / 2 - r) * rk.row_pitch
        ru = rk.panel_height
        strip_h = (ru / 2 - H / 2) / 2
        out.append(_box(f"panel{r}_top", (t / 2, 0, zr + H / 2 + strip_h), (t / 2, hw, strip_h), cls="port"))
        out.append(_box(f"panel{r}_bot", (t / 2, 0, zr - H / 2 - strip_h), (t / 2, hw, strip_h), cls="port"))
        ys = sorted(port_yz(cfg, r * rk.cols + cc)[0] for cc in range(rk.cols))
        edges = [-hw] + sum([[y - W / 2, y + W / 2] for y in ys], []) + [hw]
        for i in range(0, len(edges), 2):
            y0, y1 = edges[i], edges[i + 1]
            out.append(_box(f"panel{r}_strip{i // 2}", (t / 2, (y0 + y1) / 2, zr), (t / 2, (y1 - y0) / 2, H / 2), cls="port"))
        # number label strip above the row
        lab = T.port_labels(rk.cols, r * rk.cols + 1)
        assets.append(f'<texture name="labels{r}" type="2d" file="{lab}"/><material name="labels{r}" texture="labels{r}" specular="0.1"/>')
        out.append(_front(f"labels{r}", (-0.0004, 0, zr + H / 2 + 0.0105), (rk.cols * rk.port_pitch / 2, 0.0035), f"labels{r}"))
        for cc in range(rk.cols):
            k = r * rk.cols + cc
            y, z = port_yz(cfg, k)
            g = f"port{k}"
            zt, zb, yl, yr = z + H / 2, z - H / 2, y - W / 2, y + W / 2
            if k in act:
                # Each side of the jack = ONE convex wedge (45-degree lead-in bevel + channel wall), so
                # the plug edge never meets an internal corner between separate chamfer/wall/face boxes.
                out.append(f'<geom name="{g}_bot" type="mesh" mesh="jack_wedge_h" class="port" pos="{fmt((0, y, zb))}"/>')
                out.append(f'<geom name="{g}_top" type="mesh" mesh="jack_wedge_h" class="port" pos="{fmt((0, y, zt))}" quat="0 1 0 0"/>')
                out.append(f'<geom name="{g}_r" type="mesh" mesh="jack_wedge_v" class="port" pos="{fmt((0, yr, z))}" quat="{s2} {s2} 0 0"/>')
                out.append(f'<geom name="{g}_l" type="mesh" mesh="jack_wedge_v" class="port" pos="{fmt((0, yl, z))}" quat="{s2} {-s2} 0 0"/>')
                out.append(_box(f"{g}_back", (D + w / 2, y, z), (w / 2, W / 2 + w + c, H / 2 + w + c), cls="port"))
            else:
                out.append(_box(f"{g}_hole", (t + 0.008, y, z), (0.008, W / 2, H / 2), cls="visual", rgba=(0.02, 0.02, 0.025, 1)))
            # gold contacts at the back of the jack, and the target-marker LED (lit for the target port)
            out.append(_box(f"{g}_pins", (D - 0.0005 if k in act else t + 0.015, y, z + H / 2 - 0.0015), (0.0004, W * 0.35, 0.001), cls="visual", rgba=(0.8, 0.65, 0.2, 1)))
            out.append(_box(f"{g}_led", (-0.0006, y + 0.004, zt + 0.0035), (0.0006, 0.0015, 0.0011), cls="visual", rgba=(0.06, 0.07, 0.08, 1)))
            out.append(f'<site name="{g}" pos="{fmt((0, y, z))}" size="0.002" rgba="0 1 0 0.3" group="4"/>')
            # Occupant: an already-plugged patch cable (moved away when the port is free).
            cls = "occ" if k in act else "visual"
            out.append(f'<body name="occ{k}" pos="{fmt((0, y, z))}">')
            seat = p.seat_depth
            out.append(_box(f"occ{k}_body", (seat - pl.body_len / 2, 0, 0), (pl.body_len / 2, pl.body_w / 2, pl.body_h / 2), cls=cls, rgba=(0.85, 0.9, 0.95, 0.85)))
            out.append(_box(f"occ{k}_boot", (seat - pl.body_len - pl.boot_len / 2, 0, 0), (pl.boot_len / 2, pl.boot_w / 2, pl.boot_h / 2), cls=cls, rgba=(0.2, 0.4, 0.8, 1)))
            stub = _STUB + np.array([seat - tip_len + 0.022, 0, 0]) + rng.normal(0, 0.003, _STUB.shape)
            stub[0] = [seat - tip_len, 0, 0]  # starts at the boot rear
            for i in range(len(stub) - 1):
                out.append(f'<geom name="occ{k}_c{i}" type="capsule" class="visual" fromto="{fmt(stub[i])} {fmt(stub[i + 1])}" size="{cfg.cable.radius}" rgba="0.2 0.4 0.8 1"/>')
            out.append("</body>")
    return "\n".join(out), assets


def _rack_furniture_xml(cfg: EnvCfg):
    """Rack posts, cable managers, switch, servers, blanking panels (mostly visual)."""
    rk = cfg.rack
    hw = rk.panel_half_width
    ru = rk.panel_height
    out, assets = [], []
    floor_z = -rk.row_z  # floor in the rack frame
    top_rows = rk.rows / 2 * ru
    # posts + rails
    rail = T.rack_rail()
    assets.append(f'<texture name="rail" type="2d" file="{rail}"/><material name="rail" texture="rail" texrepeat="1 3" specular="0.4"/>')
    for sgn, nm in ((1, "l"), (-1, "r")):
        y = sgn * (hw - 0.008)
        out.append(_box(f"post_{nm}", (0.013, y, (floor_z + 0.95) / 2), (0.01, 0.008, (0.95 - floor_z) / 2), cls="visual", rgba=(0.1, 0.1, 0.11, 1)))
        out.append(_front(f"rail_{nm}", (-0.0004, y, (floor_z + 0.95) / 2), (0.008, (0.95 - floor_z) / 2), "rail"))
        out.append(_box(f"side_{nm}", (0.5, sgn * (hw + 0.04), (floor_z + 0.97) / 2), (0.5, 0.01, (0.97 - floor_z) / 2), cls="visual", rgba=(0.09, 0.09, 0.1, 1)))
    out.append(_box("rack_top", (0.5, 0, 0.98), (0.5, hw + 0.05, 0.02), cls="visual", rgba=(0.09, 0.09, 0.1, 1)))
    out.append(_box("rack_rear", (0.98, 0, (floor_z + 0.97) / 2), (0.01, hw + 0.04, (0.97 - floor_z) / 2), cls="visual", rgba=(0.05, 0.05, 0.06, 1)))
    # equipment stack: (kind, units) from just below the patch panels downwards / from above upwards
    below = [("manager", 1), ("switch", 1), ("blank", 1), ("server", 2), ("server", 1), ("server", 1), ("blank", 1), ("server", 2), ("server", 2)]
    above = [("gap", 3), ("switch", 1), ("blank", 1), ("server", 1), ("server", 2), ("server", 2), ("blank", 2), ("server", 2), ("server", 1)]
    sw = T.switch_front(48, 0)
    assets.append(f'<texture name="switch" type="2d" file="{sw}"/><material name="switch" texture="switch" specular="0.3"/>')
    bl = T.blanking_panel()
    assets.append(f'<texture name="blank" type="2d" file="{bl}"/><material name="blank" texture="blank"/>')
    n_srv = 0

    def unit(kind, units, z_center, tag):
        nonlocal n_srv
        h = units * ru / 2
        if kind == "gap":
            return
        if kind == "manager":
            out.append(_box(f"{tag}_plate", (0.002, 0, z_center), (0.002, hw - 0.02, h * 0.9), cls="visual", rgba=(0.07, 0.07, 0.08, 1)))
            for i, yy in enumerate(np.linspace(-hw + 0.05, hw - 0.05, 7)):
                out.append(_box(f"{tag}_ring{i}", (-0.03, yy, z_center), (0.03, 0.004, h * 0.8), cls="visual", rgba=(0.12, 0.12, 0.13, 1)))
            return
        out.append(_box(f"{tag}_box", (0.25, 0, z_center), (0.25, hw - 0.02, h * 0.98), cls="visual", rgba=(0.12, 0.12, 0.13, 1)))
        if kind == "switch":
            out.append(_front(f"{tag}_front", (-0.0004, 0, z_center), (hw - 0.02, h * 0.95), "switch"))
        elif kind == "blank":
            out.append(_front(f"{tag}_front", (-0.0004, 0, z_center), (hw - 0.02, h * 0.95), "blank"))
        else:
            tex = T.server_bezel(n_srv, units)
            assets.append(f'<texture name="srv{n_srv}" type="2d" file="{tex}"/><material name="srv{n_srv}" texture="srv{n_srv}" specular="0.3"/>')
            out.append(_front(f"{tag}_front", (-0.0004, 0, z_center), (hw - 0.02, h * 0.95), f"srv{n_srv}"))
            n_srv += 1

    z = -top_rows
    for i, (kind, u) in enumerate(below):
        if z - u * ru < floor_z + 0.1:
            break
        unit(kind, u, z - u * ru / 2, f"below{i}")
        z -= u * ru
    z = top_rows
    for i, (kind, u) in enumerate(above):
        if z + u * ru > 0.95:
            break
        unit(kind, u, z + u * ru / 2, f"above{i}")
        z += u * ru
    return "\n".join(out), assets


def _background_xml(cfg: EnvCfg):
    """Neighbouring racks, aisle, ceiling with light fixtures and a cable tray (all visual)."""
    if not cfg.background:
        return "", []
    rk = cfg.rack
    out, assets = [], []
    fx = rk.face_x
    for sgn, nm in ((1, "l"), (-1, "r")):
        for j in range(1, 5):
            y = sgn * (2 * rk.panel_half_width + 0.12) * j
            out.append(_box(f"nrack_{nm}{j}", (fx + 0.5, y, 1.0), (0.5, rk.panel_half_width + 0.04, 1.0), cls="visual", rgba=(0.08, 0.08, 0.09, 1)))
            for u in range(7):
                tex = T.server_bezel(100 + 10 * (j % 3) + u + (50 if sgn > 0 else 0), 2)
                mn = f"nsrv_{nm}{j}_{u}"
                assets.append(f'<texture name="{mn}" type="2d" file="{tex}"/><material name="{mn}" texture="{mn}"/>')
                out.append(f'<geom name="{mn}" type="plane" class="visual" pos="{fmt((fx - 0.001, y, 0.25 + u * 0.2))}" size="{fmt((rk.panel_half_width - 0.02, 0.085))} 0.5" quat="{fmt(FRONT_QUAT)}" material="{mn}"/>')
    # back wall behind the robot and ceiling
    out.append(_box("wall_back", (-1.6, 0, 1.5), (0.02, 3, 1.5), cls="visual", rgba=(0.55, 0.57, 0.6, 1)))
    out.append(_box("ceiling", (0.0, 0, 3.0), (3, 3, 0.02), cls="visual", rgba=(0.75, 0.76, 0.78, 1)))
    for i, x in enumerate((-0.4, 0.6)):
        out.append(_box(f"fixture{i}", (x, 0, 2.97), (0.15, 0.6, 0.02), cls="visual", rgba=(1, 1, 0.97, 1)))
    out.append(_box("tray", (fx + 0.3, 0, 2.35), (0.15, 3, 0.03), cls="visual", rgba=(0.6, 0.62, 0.15, 1)))
    for i, yy in enumerate(np.linspace(-0.5, 0.5, 6)):
        out.append(f'<geom name="tray_cable{i}" type="capsule" class="visual" fromto="{fmt((fx + 0.25, yy, 2.39))} {fmt((fx + 0.25 + 0.02 * i, yy + 0.4, 2.39))}" size="0.006" rgba="{fmt(list(CABLE_COLORS.values())[i % len(CABLE_COLORS)], 3)}"/>')
    return "\n".join(out), assets


def _jack_assets(cfg: EnvCfg) -> str:
    """Wedge meshes for the jack sides. Local frame: origin on the opening edge at the panel face,
    +x = insertion direction, +z = into the opening, extruded along y."""
    p, pl = cfg.port, cfg.plug
    W, H = pl.body_w + p.clearance, pl.body_h + p.clearance
    c, D = p.chamfer, p.depth
    th = max(p.wall, c + 0.001)
    out = []
    for name, L in (("jack_wedge_h", W + 2 * (c + 0.001)), ("jack_wedge_v", H + 2 * (c + 0.001))):
        v = []
        for yy in (-L / 2, L / 2):
            v += [(-c, yy, -th), (-c, yy, -c), (0, yy, 0), (D, yy, 0), (D, yy, -th)]
        out.append(f'<mesh name="{name}" vertex="{" ".join(fmt(x) for x in v)}"/>')
    return "\n    ".join(out)


# ---------------------------------------------------------------------- cable, robot
def _cable_xml(cfg: EnvCfg) -> str:
    c, pl = cfg.cable, cfg.plug
    m_link = c.mass_per_m * c.link_len
    L = c.link_len
    j = f'type="ball" stiffness="{c.bend_stiffness}" damping="{c.damping}" armature="{c.armature}"'
    xml, close = [], []
    for i in range(c.n_links):
        pos = "0 0 0" if i == 0 else f"{L} 0 0"
        xml.append(f'<body name="cable{i}" pos="{pos}">')
        xml.append(f'<joint name="cable{i}" {j}/>')
        xml.append(f'<geom name="cable{i}" class="cable" type="capsule" fromto="0 0 0 {L} 0 0" size="{c.radius}" mass="{m_link:.6g}"/>')
        close.append("</body>")
    # Plug (rear at the cable end, +x = insertion direction, +z = latch side).
    bl, bw, bh = pl.boot_len, pl.boot_w, pl.boot_h
    tip = pl.boot_len + pl.body_len
    xml.append(f'<body name="plug" pos="{L} 0 0">')
    xml.append(f'<joint name="plug" type="ball" stiffness="{pl.joint_stiffness}" damping="{pl.joint_damping}" armature="{c.armature}"/>')
    xml.append(f'<inertial pos="{tip / 2} 0 0" mass="{pl.mass}" diaginertia="{fmt(_box_inertia(pl.mass, (tip, pl.body_w, pl.body_h)))}"/>')
    xml.append(_box("plug_boot", (bl / 2, 0, 0), (bl / 2, bw / 2, bh / 2), cls="plug", extra='material="boot"'))
    xml.append(_box("plug_body", (bl + pl.body_len / 2, 0, 0), (pl.body_len / 2, pl.body_w / 2, pl.body_h / 2), cls="plug", extra='material="plugbody"'))
    # Visual-only details: latch tab, gold contacts, boot ribs.
    xml.append(_box("plug_latch", (bl + 0.009, 0, pl.body_h / 2 + 0.0012), (0.008, 0.003, 0.0006), cls="visual", euler=(0, -0.1, 0), extra='material="plugbody"'))
    xml.append(_box("plug_pins", (tip - 0.002, 0, -pl.body_h / 2 + 0.0014), (0.0021, pl.body_w * 0.35, 0.0012), cls="visual", rgba=(0.85, 0.7, 0.2, 1)))
    for i in range(3):
        xml.append(_box(f"plug_rib{i}", (0.003 + i * 0.004, 0, 0), (0.0008, bw / 2 + 0.0004, bh / 2 + 0.0004), cls="visual", extra='material="boot"'))
    xml.append(f'<site name="plug_tip" pos="{tip} 0 0" size="0.0015" rgba="1 0 0 1" group="4"/>')
    xml.append(f'<site name="plug_grasp" pos="{tip - pl.grasp_from_tip} 0 0" size="0.0015" rgba="0 0 1 1" group="4"/>')
    xml.append("</body>")
    xml.extend(close)
    return "\n".join(xml)


def _box_inertia(m, full):
    a, b, c = full
    return np.array([b * b + c * c, a * a + c * c, a * a + b * b]) * m / 12


def _arm_xml(cfg: EnvCfg, side: str) -> str:
    a, g = cfg.arm, cfg.gripper
    sy = ARM_SIGN[side]
    p = side[0]  # 'r' / 'l'
    out = [f'<body name="{p}_link0" pos="{a.base_x} {sy * a.base_abs_y} {a.base_z}" childclass="arm">']
    out.append(f'<geom name="{p}_link0" type="capsule" fromto="0 0 0.02 0 0 0.2" size="0.06" mass="4"/>')
    close = ["</body>"]
    for i, ((pos, quat), (lo, hi), geoms, mass) in enumerate(zip(_PANDA_CHAIN, _PANDA_RANGE, _PANDA_GEOMS, _PANDA_MASS)):
        n = f"{p}_link{i + 1}"
        out.append(f'<body name="{n}" pos="{fmt(pos)}" quat="{fmt(quat)}" gravcomp="1">')
        out.append(f'<joint name="{p}_j{i + 1}" range="{lo} {hi}"/>')
        for gi, (f, t, r) in enumerate(geoms):
            out.append(f'<geom name="{n}_g{gi}" type="capsule" fromto="{fmt(f)} {fmt(t)}" size="{r}" mass="{mass / len(geoms)}"/>')
        # joint "caps" (visual): dark rings at each joint for a more robot-like look
        out.append(f'<geom name="{n}_cap" type="cylinder" class="visual" size="{geoms[0][2] + 0.003} 0.012" rgba="0.15 0.15 0.17 1"/>')
        close.append("</body>")
    # Hand on the flange.
    fl, th = g.finger_len, g.finger_thickness
    tcp_z = 0.05 + fl - g.tcp_above_tips
    out.append(f'<body name="{p}_hand" pos="0 0 0.107" gravcomp="1">')
    out.append(f'<geom name="{p}_palm" type="box" class="gripper" pos="0 0 0.025" size="0.010 0.050 0.025" mass="0.5" material="gripper"/>')
    out.append(f'<geom name="{p}_flange" type="cylinder" class="visual" pos="0 0 -0.006" size="0.035 0.006" rgba="0.15 0.15 0.17 1"/>')
    out.append(f'<site name="{p}_tcp" pos="0 0 {tcp_z}" size="0.004" rgba="1 0.5 0 1" group="4"/>')
    # Gripper cameras: "front" sits on the hand's +x side, "rear" on its -x side, both looking at the fingertips.
    tcp = np.array([0, 0, tcp_z])
    for nm, px in (("front", 0.038), ("rear", -0.038)):
        cp = np.array([px, 0.0, 0.01])
        xy = lookat_xyaxes(cp, tcp + np.array([0, 0, 0.015]), up=(1, 0, 0))
        out.append(f'<camera name="{side}_gripper_{nm}" pos="{fmt(cp)}" xyaxes="{fmt(xy)}" fovy="80"/>')
        out.append(f'<geom name="{p}_cam_{nm}" type="box" class="visual" pos="{fmt(cp)}" size="0.008 0.012 0.006" rgba="0.1 0.1 0.1 1"/>')
        out.append(f'<geom name="{p}_lens_{nm}" type="cylinder" class="visual" pos="{fmt(cp + np.array([0, 0, 0.006]))}" size="0.004 0.001" rgba="0.02 0.05 0.1 1"/>')
    for fn, ax in (("r", 1), ("l", -1)):
        out.append(f'<body name="{p}_finger_{fn}" pos="0 0 0.05" gravcomp="1">')
        out.append(f'<joint name="{p}_finger_{fn}" type="slide" axis="0 {ax} 0" range="0 {g.max_opening}" damping="5" armature="0.01" limited="true"/>')
        # Gripping pad at the tip (20 mm); above it the finger steps 2.5 mm outwards so it clears the
        # boot of a plug in the row above when inserting into a dense panel.
        pad_l, step = 0.020, 0.0025
        out.append(f'<geom name="{p}_pad_{fn}" type="box" class="pad" pos="0 {ax * th / 2} {fl - pad_l / 2}" size="0.007 {th / 2} {pad_l / 2}" mass="0.01" material="pad"/>')
        out.append(f'<geom name="{p}_fbody_{fn}" type="box" class="gripper" pos="0 {ax * (th / 2 + step)} {(fl - pad_l) / 2}" size="0.007 {th / 2} {(fl - pad_l) / 2}" mass="0.02" material="gripper"/>')
        out.append(f'<geom name="{p}_fstep_{fn}" type="box" class="visual" pos="0 {ax * (th + step) / 2} {fl - pad_l}" size="0.007 {(th + step) / 2} 0.0015" material="gripper"/>')
        out.append("</body>")
    out.append("</body>")  # hand
    out.extend(close)
    return "\n".join(out)


def _robot_body_xml(cfg: EnvCfg) -> str:
    a = cfg.arm
    out = [
        _box("pedestal", (-0.04, 0, a.base_z / 2), (0.11, 0.40, a.base_z / 2), extra='material="robot"'),
        _box("pedestal_stripe", (0.072, 0, a.base_z - 0.06), (0.001, 0.38, 0.012), cls="visual", rgba=(0.95, 0.55, 0.1, 1)),
        f'<geom name="torso" type="capsule" class="world" fromto="-0.21 0 {a.base_z + 0.1} -0.21 0 {a.base_z + 0.58}" size="0.1" material="robot"/>',
        _box("shoulders", (-0.12, 0, a.base_z + 0.02), (0.06, 0.33, 0.03), cls="visual", extra='material="robot"'),
        _box("chest_plate", (-0.105, 0, a.base_z + 0.42), (0.008, 0.075, 0.11), cls="visual", rgba=(0.2, 0.21, 0.24, 1)),
        _box("chest_logo", (-0.096, 0, a.base_z + 0.47), (0.002, 0.04, 0.008), cls="visual", rgba=(0.95, 0.55, 0.1, 1)),
        f'<geom name="neck" type="cylinder" class="visual" pos="-0.19 0 {a.base_z + 0.66}" size="0.035 0.04" rgba="0.2 0.21 0.24 1"/>',
        _box("head", (-0.17, 0, a.base_z + 0.76), (0.07, 0.09, 0.08), extra='material="robot"'),
        _box("visor", (-0.099, 0, a.base_z + 0.78), (0.002, 0.075, 0.03), cls="visual", rgba=(0.05, 0.06, 0.08, 1)),
        _box("ear_l", (-0.17, 0.094, a.base_z + 0.78), (0.03, 0.006, 0.03), cls="visual", rgba=(0.2, 0.21, 0.24, 1)),
        _box("ear_r", (-0.17, -0.094, a.base_z + 0.78), (0.03, 0.006, 0.03), cls="visual", rgba=(0.2, 0.21, 0.24, 1)),
    ]
    # Head/chest cameras (fixed to the torso; the real base is parked during plugging).
    tgt = np.array([0.40, 0.0, 0.92])
    cams = {
        "head_front": (np.array([-0.09, 0.0, a.base_z + 0.80]), tgt, 65),
        "chest_left": (np.array([-0.13, 0.12, a.base_z + 0.45]), tgt + np.array([0, 0.05, 0]), 70),
        "chest_right": (np.array([-0.13, -0.12, a.base_z + 0.45]), tgt + np.array([0, -0.05, 0]), 70),
    }
    for n, (pos, t, fov) in cams.items():
        out.append(f'<camera name="{n}" pos="{fmt(pos)}" xyaxes="{fmt(lookat_xyaxes(pos, t))}" fovy="{fov}"/>')
    return "\n".join(out)


def _clutter_xml(cfg: EnvCfg) -> str:
    """Pre-allocated loose objects (parked off-scene unless a scenario places them on the shelf)."""
    shapes = [
        # screwdriver: handle + shaft
        '<geom type="capsule" size="0.011 0.035" euler="0 1.5708 0" rgba="0.85 0.15 0.1 1" mass="0.05"/>'
        '<geom type="cylinder" size="0.003 0.04" pos="0.075 0 0" euler="0 1.5708 0" rgba="0.7 0.7 0.72 1" mass="0.02"/>',
        '<geom type="cylinder" size="0.03 0.012" rgba="0.1 0.1 0.1 1" mass="0.08"/>',  # tape roll
        '<geom type="box" size="0.05 0.03 0.015" rgba="0.95 0.85 0.2 1" mass="0.15"/>',  # label printer
        '<geom type="box" size="0.04 0.025 0.004" rgba="0.9 0.9 0.92 0.8" mass="0.02"/>',  # bag of cable ties
        '<geom type="box" size="0.02 0.006 0.004" rgba="0.85 0.9 0.95 0.9" mass="0.005"/>',  # spare plug
        '<geom type="cylinder" size="0.045 0.008" rgba="0.2 0.4 0.85 1" mass="0.1"/>',  # coiled cable
    ]
    out = []
    for i in range(cfg.n_clutter):
        parts = shapes[i % len(shapes)].split("<geom ")[1:]
        geoms = "".join(f'<geom name="clutter{i}_g{j}" class="world" ' + g for j, g in enumerate(parts))
        out.append(f'<body name="clutter{i}" pos="{3 + i} 3 0.05"><freejoint name="clutter{i}"/>{geoms}</body>')
    # visual-only loose cable (static body, geoms re-posed per episode)
    out.append('<body name="loose_cable" pos="0 0 -5">')
    for i in range(cfg.loose_cable_links):
        out.append(f'<geom name="loose{i}" type="capsule" class="visual" size="{cfg.cable.radius} 0.015" rgba="0.6 0.6 0.6 1"/>')
    out.append("</body>")
    return "\n".join(out)


def build_mjcf(cfg: EnvCfg) -> str:
    s, cab, rk = cfg.shelf, cfg.cable, cfg.rack
    shelf_c = ((s.x_min + s.x_max) / 2, 0, s.top_z - 0.01)
    shelf_h = ((s.x_max - s.x_min) / 2, s.y_half, 0.01)
    anchor_z = s.top_z + cfg.plug.body_h / 2
    acts, eqs = [], []
    for side in ARMS:
        p = side[0]
        for i, (lo, hi) in enumerate(_PANDA_RANGE):
            acts.append(
                f'<position name="{p}_j{i + 1}" joint="{p}_j{i + 1}" kp="{cfg.arm.kp[i]}" kv="{cfg.arm.kv[i]}" '
                f'ctrlrange="{lo} {hi}" forcerange="{-_PANDA_FORCE[i]} {_PANDA_FORCE[i]}"/>'
            )
        F = cfg.gripper.max_force
        acts.append(f'<general name="{p}_grip" joint="{p}_finger_r" ctrlrange="0 1" gainprm="{-2 * F}" biasprm="{F} 0 -20" biastype="affine" forcerange="{-F} {F}"/>')
        eqs.append(f'<joint name="{p}_finger_couple" joint1="{p}_finger_l" joint2="{p}_finger_r" polycoef="0 1 0 0 0" solref="0.005 1"/>')
    eqs.append('<weld name="latch" body1="latch_anchor" body2="plug" active="false" solref="0.01 1"/>')
    panel, a1 = _panel_xml(cfg)
    furn, a2 = _rack_furniture_xml(cfg)
    bg, a3 = _background_xml(cfg)
    tile, mat = T.floor_tile(), T.esd_mat()
    w, h = cfg.camera.width, cfg.camera.height
    return f"""<mujoco model="robot_plugging">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.timestep}" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <size memory="96M"/>
  <visual>
    <global offwidth="{max(w, 1280)}" offheight="{max(h, 720)}"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.22 0.22 0.22" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <map znear="0.005" zfar="20"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.8 0.85" rgb2="0.35 0.4 0.45" width="256" height="256"/>
    <texture name="floor" type="2d" file="{tile}"/>
    <material name="floor" texture="floor" texrepeat="1.67 1.67" texuniform="true" reflectance="0.05"/>
    <texture name="esd" type="2d" file="{mat}"/>
    <material name="esd" texture="esd" texrepeat="2 2"/>
    <material name="shelf" rgba="0.45 0.47 0.5 1" specular="0.1"/>
    <material name="panel" rgba="0.08 0.08 0.09 1" specular="0.3"/>
    <material name="rack" rgba="0.15 0.15 0.17 1" specular="0.4"/>
    <material name="robot" rgba="0.85 0.86 0.88 1" specular="0.3"/>
    <material name="armmat" rgba="0.92 0.92 0.93 1" specular="0.4"/>
    <material name="gripper" rgba="0.2 0.2 0.22 1"/>
    <material name="pad" rgba="0.1 0.1 0.1 1"/>
    <material name="plugbody" rgba="0.85 0.9 0.95 0.85" specular="0.6"/>
    <material name="boot" rgba="0.15 0.35 0.85 1"/>
    {_jack_assets(cfg)}
    {"".join(a1 + a2 + a3)}
  </asset>
  <default>
    <joint axis="0 0 1"/>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="world"><geom contype="1" conaffinity="7" friction="0.8 0.01 0.001"/></default>
    <default class="port"><geom type="box" contype="1" conaffinity="7" friction="{cfg.port.friction} 0.005 0.0005" material="panel" solimp="{cfg.port.solimp}" solref="{cfg.port.solref} 1"/></default>
    <default class="occ"><geom type="box" contype="1" conaffinity="7" friction="{cfg.plug.friction} 0.005 0.0005"/></default>
    <default class="plug"><geom contype="1" conaffinity="7" condim="4" friction="{cfg.plug.friction} 0.005 0.0005" solimp="{cfg.port.solimp}" solref="{cfg.port.solref} 1"/></default>
    <default class="cable"><geom contype="2" conaffinity="1" condim="3" friction="{cab.friction} 0.005 0.0005" rgba="{fmt(CABLE_COLORS['blue'], 3)}"/></default>
    <default class="gripper"><geom contype="1" conaffinity="7"/></default>
    <default class="pad"><geom contype="1" conaffinity="7" condim="4" friction="{cfg.gripper.pad_friction} 0.02 0.002"/></default>
    <default class="visual"><geom contype="0" conaffinity="0" group="2"/></default>
    <default class="arm"><joint armature="0.1" damping="1"/><geom contype="4" conaffinity="1" material="armmat"/></default>
  </default>
  <worldbody>
    <light name="key" pos="0.3 0 2.4" dir="0.1 0 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="-0.5 0.8 1.8" dir="0.5 -0.6 -0.6" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <light name="ceil_a" pos="-0.4 0 2.9" dir="0.2 0 -1" diffuse="0.2 0.2 0.2" castshadow="false"/>
    <light name="ceil_b" pos="0.6 0 2.9" dir="-0.1 0 -1" diffuse="0.2 0.2 0.2" castshadow="false"/>
    <light name="spot" pos="0.0 -0.6 1.6" dir="0.6 0.6 -0.5" diffuse="0 0 0" cutoff="20" exponent="20" castshadow="true"/>
    <light name="back" pos="{rk.face_x + 0.6} 0 1.4" dir="-1 0 -0.2" diffuse="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 1" material="floor" class="world"/>
    <body name="latch_anchor" pos="0 0 0"/>
    <geom name="shelf" type="box" class="world" pos="{fmt(shelf_c)}" size="{fmt(shelf_h)}" material="shelf" friction="{s.friction} 0.01 0.001"/>
    <geom name="shelf_mat" type="box" class="visual" pos="{fmt((shelf_c[0], 0, s.top_z - 0.0004))}" size="{fmt((shelf_h[0] - 0.005, s.y_half - 0.005, 0.0005))}" material="esd"/>
    <geom name="shelf_cabinet" type="box" class="visual" pos="{fmt((shelf_c[0], 0, (s.top_z - 0.02) / 2))}" size="{fmt((shelf_h[0] - 0.01, s.y_half - 0.01, (s.top_z - 0.02) / 2))}" material="rack"/>
    <body name="rack" pos="{rk.face_x} 0 {rk.row_z}">
      {panel}
      {furn}
    </body>
    {bg}
    {_robot_body_xml(cfg)}
    {_arm_xml(cfg, "right")}
    {_arm_xml(cfg, "left")}
    <body name="switch" pos="{cab.anchor_x} {-cab.anchor_abs_y} {anchor_z}">
      <geom name="switch" type="box" class="world" pos="-0.032 0 0.011" size="0.03 0.05 0.015" rgba="0.12 0.12 0.14 1"/>
      <geom name="switch_led" type="box" class="visual" pos="-0.0015 0.02 0.018" size="0.001 0.003 0.002" rgba="0.1 0.9 0.2 1"/>
      {_cable_xml(cfg)}
    </body>
    {_clutter_xml(cfg)}
  </worldbody>
  <contact>
    <exclude body1="r_hand" body2="r_link6"/><exclude body1="r_hand" body2="r_link5"/>
    <exclude body1="l_hand" body2="l_link6"/><exclude body1="l_hand" body2="l_link5"/>
  </contact>
  <equality>
    {"".join(eqs)}
  </equality>
  <actuator>
    {"".join(acts)}
  </actuator>
</mujoco>
"""
