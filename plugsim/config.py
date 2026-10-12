"""Every modelling assumption lives here, as a dataclass field with a default.

Nothing in this file is measured on the real WR1. These are deliberately strong placeholder
assumptions (see ASSUMPTIONS.md) so we can build the full pipeline now and swap in real numbers later.
All lengths in metres, angles in radians, masses in kg, forces in N.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class PlugCfg:
    # RJ45 8P8C plug body (the part that enters the jack). Real: ~11.7 x 8.0 x 21 mm.
    body_len: float = 0.021
    body_w: float = 0.0117
    body_h: float = 0.0080
    # Strain-relief boot behind the body (where we grasp).
    boot_len: float = 0.016
    boot_w: float = 0.0100
    boot_h: float = 0.0076
    mass: float = 0.012
    # Where the gripper grasps, measured backwards from the plug tip.
    grasp_from_tip: float = 0.031
    # Plastic. MuJoCo uses the max of the two geoms' friction, so keep this low: the pads (1.5) still
    # govern the grasp, while plug-jack contact stays at ~0.3 (otherwise the chamfer self-locks).
    friction: float = 0.3
    # Bending stiffness/damping of the joint between plug and last cable link.
    joint_stiffness: float = 0.02
    joint_damping: float = 0.002


@dataclass
class PortCfg:
    # Total clearance per dimension (jack opening = plug + clearance). Real RJ45: ~0.2-0.5 mm.
    # Kept loose by default for stable contacts; tighten in notebook 01.
    clearance: float = 0.0010
    depth: float = 0.020  # channel depth until the back stop
    seat_depth: float = 0.0150  # tip depth at which the latch "clicks"
    chamfer: float = 0.0015  # entrance chamfer (per side)
    wall: float = 0.0020
    friction: float = 0.3
    # Contact parameters for plug<->jack contacts. Sharper than the global default so the sub-mm
    # clearances and the chamfer are not swallowed by contact softness.
    solref: float = 0.004
    solimp: str = "0.99 0.999 0.0001"
    # Latch tab detent: resistance ramping up to `latch_force` between `latch_engage` and `seat_depth`
    # (real RJ45 insertion takes ~10-20 N). A plug released before the click is pushed back out.
    latch_force: float = 10.0
    latch_engage: float = 0.010
    # Latch only engages if the plug is this well aligned at seat depth.
    latch_max_lateral: float = 0.0012
    latch_max_angle: float = 0.09  # ~5 deg


@dataclass
class RackCfg:
    # A 19" rack with `rows` stacked 1U 24-port patch panels facing the robot (insertion along +x).
    # Ports with |y| <= active_half_width are physical jacks (and possible targets); the outer ones are
    # visual-only to keep collision cost down (the arms never go there).
    rows: int = 2
    cols: int = 24
    port_pitch: float = 0.0185
    row_pitch: float = 0.04445  # 1U
    active_half_width: float = 0.12
    face_x: float = 0.53  # nominal x of the panel front face
    row_z: float = 1.05  # nominal height of the centre between the panel rows
    panel_thickness: float = 0.003
    panel_height: float = 0.0445  # 1U
    panel_half_width: float = 0.2413  # 19" = 482.6 mm incl. ears

    @property
    def n_ports(self) -> int:
        return self.rows * self.cols


@dataclass
class ShelfCfg:
    # Shelf/tray in front of the rack where the cable lies.
    x_min: float = 0.15
    x_max: float = 0.51
    y_half: float = 0.42
    top_z: float = 0.80
    friction: float = 0.9


@dataclass
class CableCfg:
    # Cat6 patch cable, modelled as a chain of capsules with ball joints (MJX/Warp-friendly, no plugins).
    # The far end is plugged into a "switch" on the shelf (fixed anchor); the plug end lies free.
    n_links: int = 20
    link_len: float = 0.035  # total length = n_links * link_len = 0.70 m
    radius: float = 0.003
    mass_per_m: float = 0.045
    bend_stiffness: float = 0.004  # N m / rad per joint
    damping: float = 0.0008
    armature: float = 2e-6
    friction: float = 0.8
    # Switch (anchor) position: a side of the shelf near the rack, side chosen per episode.
    anchor_x: float = 0.44
    anchor_abs_y: float = 0.34


@dataclass
class ArmCfg:
    # WR1 proxy: two 7-DoF arms with Franka-Panda-like kinematics, built from primitives.
    base_x: float = 0.0
    base_abs_y: float = 0.26
    base_z: float = 0.80
    kp: tuple = (4500, 4500, 3500, 3500, 2000, 2000, 2000)
    kv: tuple = (200, 200, 150, 150, 60, 60, 60)
    # Right-arm home (TCP ~ (0.36, -0.25, 1.06), pointing down); the left arm uses the mirror image.
    home_q: tuple = (0.0, -0.475, 0.0, -2.764, 0.0, 2.289, 0.0)


@dataclass
class GripperCfg:
    # Parallel-jaw gripper, force controlled: ctrl 1 = close with max_force, 0 = open with max_force.
    max_force: float = 40.0
    max_opening: float = 0.040  # per finger
    finger_len: float = 0.055  # long enough that the palm clears plugs in the row above
    finger_thickness: float = 0.004  # thin enough to fit between neighbouring plugs (18.5 mm pitch)
    pad_friction: float = 1.5
    tcp_above_tips: float = 0.006  # TCP site sits this far above the fingertips
    # Compliant mode (action dim "compliant"=1): per-axis IK gains in the TCP frame (x = along the
    # plug, y = finger axis, z = approach; then rotations) acting at a fixed tool point = where the
    # plug tip sits for a nominal grasp.
    compliant_gains: tuple = (1.0, 0.1, 0.1, 0.3, 0.3, 0.3)
    tool_tip: tuple = (0.031, 0.0, 0.0035)


@dataclass
class CameraCfg:
    width: int = 384
    height: int = 216  # 16:9 like the real 1280x720 feeds
    # The real WR1 has 8 cameras: head front/back, chest L/R, 2 per gripper.
    # We model 7 (head-back is omitted) and by default record the 4 gripper cameras the paper uses.
    record: tuple = (
        "right_gripper_front",
        "right_gripper_rear",
        "left_gripper_front",
        "left_gripper_rear",
    )
    all_cameras: tuple = (
        "head_front",
        "chest_left",
        "chest_right",
        "right_gripper_front",
        "right_gripper_rear",
        "left_gripper_front",
        "left_gripper_rear",
    )


@dataclass
class EnvCfg:
    plug: PlugCfg = field(default_factory=PlugCfg)
    port: PortCfg = field(default_factory=PortCfg)
    rack: RackCfg = field(default_factory=RackCfg)
    shelf: ShelfCfg = field(default_factory=ShelfCfg)
    cable: CableCfg = field(default_factory=CableCfg)
    arm: ArmCfg = field(default_factory=ArmCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    timestep: float = 0.002
    control_hz: float = 20.0  # policy / recording rate
    max_dpos: float = 0.02  # max EE translation per control step (0.4 m/s)
    max_drot: float = 0.15  # max EE rotation per control step (rad)
    episode_max_s: float = 16.0
    settle_s: float = 0.4  # physics settling after reset before the episode clock starts
    n_clutter: int = 6  # pre-allocated clutter objects on the shelf (parked off-scene unless sampled)
    loose_cable_links: int = 16  # a visual-only second cable lying on the shelf (distractor)
    background: bool = True  # neighbouring racks, raised floor, ceiling, cable tray

    def to_dict(self):
        return dataclasses.asdict(self)
