"""PlugEnv: bimanual cable plugging in MuJoCo.

Action (16-D, control rate `cfg.control_hz`): for each arm in ARMS order (right, left)
    [dx, dy, dz, drx, dry, drz, grip, compliant]
  - TCP translation delta (world frame, m) and rotation delta (world-frame rotation vector, rad),
    applied to the arm's internal TCP *target* (not the measured pose), so replaying recorded actions
    reproduces the same target trajectory.
  - grip in [0, 1]: 1 = close with max force, 0 = open, 0.5 = zero force (force-controlled gripper).
  - compliant > 0.5: impedance-like insertion mode (stiff along the plug axis, soft laterally and in
    rotation, in the TCP frame) about the fixed tool tip `cfg.gripper.tool_tip`; otherwise stiff
    position control of the TCP.

Low level: damped-least-squares IK at every physics step -> joint position actuators (+ gravity
compensation). Contact forces are therefore bounded by the actuator stiffness.

Observation `state` (44-D): per arm q(7), qd(7), tcp_pos(3), tcp_quat(4), finger opening(1).
`privileged()` gives ground truth (plug pose, insertion depth, latch, grasp) for labels and the expert.
"""
from __future__ import annotations

import zlib

import mujoco
import numpy as np

from . import geometry as G
from .config import EnvCfg
from .scenario import LED_OFF, Scenario, sample_scenario
from .scene import ARMS, LIGHTS, active_ports, build_mjcf

Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # TCP pointing down, hand x along world +x
ACTION_FIELDS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip", "compliant")
ACT_DIM = len(ACTION_FIELDS)


def state_names():
    names = []
    for side in ARMS:
        names += [f"{side}.q{i + 1}" for i in range(7)] + [f"{side}.qd{i + 1}" for i in range(7)]
        names += [f"{side}.tcp_{a}" for a in "xyz"] + [f"{side}.tcp_q{a}" for a in "wxyz"] + [f"{side}.grip_width"]
    return names


def action_names():
    return [f"{side}.{a}" for side in ARMS for a in ACTION_FIELDS]


def apply_image_fx(img: np.ndarray, fx: dict, seed: int) -> np.ndarray:
    """Camera degradations: gain/gamma (exposure), motion blur, Gaussian + salt noise, lens occlusion."""
    if not fx:
        return img
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    if fx.get("blur", 0) > 1:
        n, a = int(fx["blur"]), fx.get("blur_angle", 0.0)
        acc = np.zeros_like(x)
        for i in range(n):
            s = i - (n - 1) / 2
            acc += np.roll(np.roll(x, int(round(s * np.cos(a))), 1), int(round(s * np.sin(a))), 0)
        x = acc / n
    if "occlusion" in fx:
        cx, cy, r = fx["occlusion"]
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((xx / w - cx) / r) ** 2 + ((yy / h - cy) / (r * 1.3)) ** 2)
        m = np.clip(1.3 - d, 0, 1)[..., None] ** 0.7
        x = x * (1 - 0.93 * m) + 0.10 * 0.93 * m
    x = x * fx.get("gain", 1.0)
    if fx.get("gamma", 1.0) != 1.0:
        x = np.clip(x, 0, None) ** fx["gamma"]
    if fx.get("noise", 0) > 0:
        x = x + rng.normal(0, fx["noise"] / 255.0, x.shape).astype(np.float32)
    if fx.get("salt", 0) > 0:
        m = rng.random(x.shape[:2]) < fx["salt"]
        x[m] = rng.choice([0.0, 1.0], size=(int(m.sum()), 1))
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class PlugEnv:
    def __init__(self, cfg: EnvCfg | None = None, render: bool = True):
        self.cfg = cfg or EnvCfg()
        self.xml = build_mjcf(self.cfg)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(1.0 / (self.cfg.control_hz * self.cfg.timestep)))
        self.dt = self.n_sub * self.cfg.timestep
        self._renderer = None
        self._render_enabled = render

        # ---- indices ----
        self.arm = {}
        for side in ARMS:
            p = side[0]
            jids = [m.joint(f"{p}_j{i + 1}").id for i in range(7)]
            self.arm[side] = dict(
                qadr=np.array([m.jnt_qposadr[j] for j in jids]),
                dadr=np.array([m.jnt_dofadr[j] for j in jids]),
                lo=m.jnt_range[jids, 0].copy(),
                hi=m.jnt_range[jids, 1].copy(),
                act=np.array([m.actuator(f"{p}_j{i + 1}").id for i in range(7)]),
                grip_act=m.actuator(f"{p}_grip").id,
                fing=[m.jnt_qposadr[m.joint(f"{p}_finger_{f}").id] for f in "rl"],
                tcp=m.site(f"{p}_tcp").id,
                hand_body=m.body(f"{p}_hand").id,
                pads={m.geom(f"{p}_pad_r").id, m.geom(f"{p}_pad_l").id},
            )
        cfg = self.cfg
        self.plug_body = m.body("plug").id
        self.plug_geoms = {m.geom("plug_boot").id, m.geom("plug_body").id}
        self.boot_geoms = [m.geom("plug_boot").id] + [m.geom(f"plug_rib{i}").id for i in range(3)]
        self.plug_tip_site = m.site("plug_tip").id
        self.plug_grasp_site = m.site("plug_grasp").id
        self.switch_body = m.body("switch").id
        self.rack_body = m.body("rack").id
        self.cable_joints = [m.joint(f"cable{i}").id for i in range(cfg.cable.n_links)] + [m.joint("plug").id]
        self.cable_geoms = [m.geom(f"cable{i}").id for i in range(cfg.cable.n_links)]
        self.cable_bodies = [m.body(f"cable{i}").id for i in range(cfg.cable.n_links)]
        self.latch_eq = m.equality("latch").id
        n = cfg.rack.n_ports
        self.active = np.array(active_ports(cfg))
        self.port_sites = [m.site(f"port{k}").id for k in range(n)]
        self._active_sites = np.array([self.port_sites[k] for k in self.active])
        self.port_leds = [m.geom(f"port{k}_led").id for k in range(n)]
        self.occ_bodies = [m.body(f"occ{k}").id for k in range(n)]
        self.occ_color_geoms = [[m.geom(f"occ{k}_boot").id] + [m.geom(f"occ{k}_c{i}").id for i in range(4)] for k in range(n)]
        self.cam_ids = {c: m.camera(c).id for c in cfg.camera.all_cameras}
        jack = ("_bot", "_top", "_l", "_r", "_back", "_hole")
        self.panel_geoms = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("panel")
                            or ((m.geom(i).name or "").startswith("port") and m.geom(i).name.endswith(jack))]
        self.clutter_qadr = [m.jnt_qposadr[m.joint(f"clutter{i}").id] for i in range(cfg.n_clutter)]
        self.loose_body = m.body("loose_cable").id
        self.loose_geoms = [m.geom(f"loose{i}").id for i in range(cfg.loose_cable_links)]
        self.light_ids = {nm: m.light(nm).id for nm in LIGHTS}
        self._orig = dict(
            geom_rgba=m.geom_rgba.copy(),
            geom_friction=m.geom_friction.copy(),
            geom_pos=m.geom_pos.copy(),
            geom_quat=m.geom_quat.copy(),
            geom_size=m.geom_size.copy(),
            body_mass=m.body_mass.copy(),
            body_inertia=m.body_inertia.copy(),
            jnt_stiffness=m.jnt_stiffness.copy(),
            light_diffuse=m.light_diffuse.copy(),
            cam_pos=m.cam_pos.copy(),
            cam_quat=m.cam_quat.copy(),
            body_pos=m.body_pos.copy(),
            body_quat=m.body_quat.copy(),
            mat_rgba=m.mat_rgba.copy(),
            actuator_gainprm=m.actuator_gainprm.copy(),
            actuator_biasprm=m.actuator_biasprm.copy(),
            actuator_forcerange=m.actuator_forcerange.copy(),
        )
        self._light_base = m.light_diffuse.copy()
        self._head_orig = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self._kv_over_kp = np.array(cfg.arm.kv, float) / np.array(cfg.arm.kp, float)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        mirror = np.array([-1, 1, -1, 1, -1, 1, -1.0])
        self.home_q = {"right": np.array(cfg.arm.home_q), "left": np.array(cfg.arm.home_q) * mirror}
        self.scenario: Scenario | None = None
        self.reset(sample_scenario(cfg, 0))

    # ------------------------------------------------------------------ reset / randomisation
    def _apply_scenario(self, sc: Scenario):
        m, o, cfg = self.model, self._orig, self.cfg
        for k, v in o.items():
            getattr(m, k)[:] = v
        # Cable + boot colour.
        for g in self.cable_geoms:
            m.geom_rgba[g] = sc.cable_color
        for g in self.boot_geoms:
            m.geom_rgba[g] = sc.boot_color
        m.mat_rgba[m.material("boot").id] = sc.boot_color
        for g in self.panel_geoms:
            m.geom_rgba[g] = sc.panel_rgba
        m.mat_rgba[m.material("panel").id] = sc.panel_rgba
        m.mat_rgba[m.material("esd").id] = (sc.mat_tint, sc.mat_tint, sc.mat_tint, 1)
        # LEDs: target marker + optional distractors.
        for led in self.port_leds:
            m.geom_rgba[led] = LED_OFF
        for k, *rgba in sc.led_distractors:
            m.geom_rgba[self.port_leds[k]] = rgba
        m.geom_rgba[self.port_leds[sc.port_idx]] = sc.target_led
        # Occupied ports: move occupant plugs in (they are parked far below the floor otherwise).
        occ = {int(e[0]): e[1:] for e in sc.occupied}
        for k, b in enumerate(self.occ_bodies):
            if k in occ:
                for g in self.occ_color_geoms[k]:
                    m.geom_rgba[g] = occ[k]
            else:
                m.body_pos[b, 2] = -20.0
        # Lights.
        for nm, lid in self.light_ids.items():
            if nm in sc.lights:
                m.light_diffuse[lid] = sc.lights[nm]
            m.light_active[lid] = float(np.sum(m.light_diffuse[lid])) > 1e-6  # dark lights cost a (shadow) pass
        self._light_base = m.light_diffuse.copy()
        hs, *ht = sc.headlight
        m.vis.headlight.ambient[:] = self._head_orig[0] * hs * np.asarray(ht)
        m.vis.headlight.diffuse[:] = self._head_orig[1] * hs * np.asarray(ht)
        self._head_base = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        for c, jit in sc.cam_jitter.items():
            cid = self.cam_ids[c]
            m.cam_pos[cid] = o["cam_pos"][cid] + np.asarray(jit[:3])
            m.cam_quat[cid] = G.quat_mul(G.quat_from_rotvec(jit[3:]), o["cam_quat"][cid])
        # Physics / hardware.
        for side, gs in zip(ARMS, sc.arm_gain_scale):
            for i, a in enumerate(self.arm[side]["act"]):
                kp = o["actuator_gainprm"][a, 0] * gs
                m.actuator_gainprm[a, 0] = kp
                m.actuator_biasprm[a, 1] = -kp
            ga = self.arm[side]["grip_act"]
            m.actuator_gainprm[ga, 0] = o["actuator_gainprm"][ga, 0] * sc.grip_force_scale
            m.actuator_biasprm[ga, 0] = o["actuator_biasprm"][ga, 0] * sc.grip_force_scale
            m.actuator_forcerange[ga] = o["actuator_forcerange"][ga] * sc.grip_force_scale
            for g in self.arm[side]["pads"]:
                m.geom_friction[g, 0] = o["geom_friction"][g, 0] * sc.pad_friction_scale
        for b in self.cable_bodies:
            m.body_mass[b] = o["body_mass"][b] * sc.cable_mass_scale
            m.body_inertia[b] = o["body_inertia"][b] * sc.cable_mass_scale
        for j in self.cable_joints[:-1]:
            m.jnt_stiffness[j] = o["jnt_stiffness"][j] * sc.cable_stiffness_scale
        # Rack pose (the base never parks identically).
        dx, dy, dz, yaw = sc.rack_pose
        m.body_pos[self.rack_body] = o["body_pos"][self.rack_body] + (dx, dy, dz)
        m.body_quat[self.rack_body] = G.yaw_quat(yaw)
        # Switch (cable anchor) pose.
        m.body_pos[self.switch_body, :2] = (sc.anchor_x, sc.anchor_side * cfg.cable.anchor_abs_y)
        m.body_quat[self.switch_body] = G.yaw_quat(sc.theta0)
        # Loose (visual) cable on the shelf.
        pts = np.asarray(sc.loose_cable.get("points", [])) if sc.loose_cable else np.zeros((0, 2))
        if len(pts) >= 2:
            m.body_pos[self.loose_body] = (0, 0, cfg.shelf.top_z + cfg.cable.radius)
            for i, g in enumerate(self.loose_geoms):
                a, b = pts[min(i, len(pts) - 2)], pts[min(i + 1, len(pts) - 1)]
                d = np.append(b - a, 0.0)
                L = np.linalg.norm(d) + 1e-9
                m.geom_pos[g] = np.append((a + b) / 2, 0.0)
                m.geom_quat[g] = G.mat_to_quat(_frame_z(d / L))
                m.geom_size[g, 1] = L / 2
                m.geom_rgba[g] = sc.loose_cable["color"]
        else:
            m.body_pos[self.loose_body] = (0, 0, -5)

    def reset(self, scenario: Scenario | None = None, seed: int | None = None):
        if scenario is None:
            scenario = sample_scenario(self.cfg, seed if seed is not None else np.random.randint(1 << 30))
        self.scenario = scenario
        m, d = self.model, self.data
        self._apply_scenario(scenario)
        mujoco.mj_resetData(m, d)
        d.eq_active[self.latch_eq] = 0
        # Arms at home (+ noise), grippers open.
        hn = np.asarray(scenario.home_noise) if scenario.home_noise else np.zeros(14)
        for i, side in enumerate(ARMS):
            a = self.arm[side]
            q = np.clip(self.home_q[side] + hn[7 * i : 7 * i + 7], a["lo"], a["hi"])
            d.qpos[a["qadr"]] = q
            d.ctrl[a["act"]] = q
            d.qpos[a["fing"]] = self.cfg.gripper.max_opening
            d.ctrl[a["grip_act"]] = 0.0
        # Cable layout: planar turn angles on the chain's ball joints.
        for j, dl in zip(self.cable_joints, list(scenario.deltas) + [scenario.plug_delta]):
            d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 4] = G.yaw_quat(dl)
        # Clutter: park off-scene unless placed.
        placed = {int(c[0]): c[1:] for c in scenario.clutter}
        for i, qa in enumerate(self.clutter_qadr):
            if i in placed:
                x, y, yaw = placed[i]
                d.qpos[qa : qa + 3] = (x, y, self.cfg.shelf.top_z + 0.02)
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
            else:
                d.qpos[qa : qa + 3] = (3 + i, 3, 0.05)
        mujoco.mj_forward(m, d)
        self.targets = {s: self._tcp_pose(s) for s in ARMS}
        self.grip_cmd = {s: 0.0 for s in ARMS}
        self.compliance = {s: np.ones(6) for s in ARMS}
        self.compliance_center = {s: None for s in ARMS}
        self.latched_port = -1
        self.latch_time = None
        self.ext_force = None  # (body, wrench(6), steps_left)
        self.unstable = False
        self.time = 0.0
        self.step_count = 0
        # Let the cable (and clutter) settle on the shelf before the episode clock starts.
        for _ in range(int(self.cfg.settle_s / self.cfg.timestep)):
            self._physics_step()
        mujoco.mj_forward(m, d)
        self.time = 0.0
        self.targets = {s: self._tcp_pose(s) for s in ARMS}
        self._update_lights()
        return self.observe()

    # ------------------------------------------------------------------ control
    def _tcp_pose(self, side):
        sid = self.arm[side]["tcp"]
        return self.data.site_xpos[sid].copy(), G.mat_to_quat(self.data.site_xmat[sid])

    def _ik(self, side, tgt_p, tgt_q, tgt_v=None):
        m, d, a = self.model, self.data, self.arm[side]
        p, q = self._tcp_pose(side)
        off = self.compliance_center[side]
        if off is None:
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, a["tcp"])
        else:
            # Control a point rigidly offset from the TCP (e.g. the plug tip): remote-centre compliance.
            p = p + G.quat_rotate(q, off)
            tgt_p = tgt_p + G.quat_rotate(tgt_q, off)
            mujoco.mj_jac(m, d, self._jacp, self._jacr, p, a["hand_body"])
        err = np.concatenate([tgt_p - p, G.quat_error(tgt_q, q)])
        gains = self.compliance[side]
        if not np.all(gains == 1.0):
            R = d.site_xmat[a["tcp"]].reshape(3, 3)  # gains act in the TCP frame
            err = np.concatenate([R @ (gains[:3] * (R.T @ err[:3])), R @ (gains[3:] * (R.T @ err[3:]))])
        J = np.vstack([self._jacp[:, a["dadr"]], self._jacr[:, a["dadr"]]])
        JJt = J @ J.T + 1e-3 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # Null-space pull towards the home posture keeps the redundant DoF well-behaved.
        qcur = d.qpos[a["qadr"]]
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq += (np.eye(7) - Jpinv @ J) @ (0.05 * (self.home_q[side] - qcur))
        # Velocity feed-forward cancels the PD actuators' lag (error ~ kv/kp * velocity).
        if tgt_v is not None:
            dq += self._kv_over_kp * (Jpinv @ tgt_v)
        dq = np.clip(dq, -0.25, 0.25)
        d.ctrl[a["act"]] = np.clip(qcur + dq, a["lo"], a["hi"])

    def _physics_step(self, control=None):
        """One physics step. mj_step1 refreshes kinematics, then `control()` (the IK) sets ctrl from
        fresh positions/Jacobians, then mj_step2 integrates. This keeps the controller's inputs a pure
        function of the state, so set_state() + step() reproduces a recording exactly."""
        m, d = self.model, self.data
        n_bad = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        mujoco.mj_step1(m, d)
        # Applied forces are set after mj_step1 (fresh kinematics) and used in mj_step2.
        d.xfrc_applied[:] = 0
        if self.ext_force is not None:
            b, w, n = self.ext_force
            d.xfrc_applied[b] += w
            self.ext_force = (b, w, n - 1) if n > 1 else None
        self._latch_detent()
        if control is not None:
            control()
        mujoco.mj_step2(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > n_bad:
            self.unstable = True  # MuJoCo has auto-reset the state; the episode must be discarded
        self._check_latch()

    def clip_action(self, action):
        """The action the env actually executes: translation/rotation deltas clipped to the limits."""
        a = np.asarray(action, float).reshape(len(ARMS), ACT_DIM).copy()
        for i in range(len(ARMS)):
            for sl, lim in ((slice(0, 3), self.cfg.max_dpos), (slice(3, 6), self.cfg.max_drot)):
                n = np.linalg.norm(a[i, sl])
                if n > lim:
                    a[i, sl] *= lim / n
        return a.reshape(-1)

    def step(self, action):
        action = self.clip_action(action).reshape(len(ARMS), ACT_DIM)
        cfg = self.cfg
        start = {s: self.targets[s] for s in ARMS}
        goal = {}
        for i, side in enumerate(ARMS):
            dp, dr = action[i, :3], action[i, 3:6]
            p0, q0 = start[side]
            goal[side] = (p0 + dp, G.quat_mul(G.quat_from_rotvec(dr), q0))
            self.grip_cmd[side] = float(np.clip(action[i, 6], 0.0, 1.0))
            self.data.ctrl[self.arm[side]["grip_act"]] = self.grip_cmd[side]
            if action[i, 7] > 0.5:
                self.set_compliance(side, cfg.gripper.compliant_gains, center=cfg.gripper.tool_tip)
            else:
                self.set_compliance(side, None)
        vel = {s: np.concatenate([goal[s][0] - start[s][0], G.quat_error(goal[s][1], start[s][1])]) / self.dt for s in ARMS}
        for k in range(self.n_sub):
            t = (k + 1) / self.n_sub

            def control(t=t):
                for side in ARMS:
                    (p0, q0), (p1, q1) = start[side], goal[side]
                    self._ik(side, p0 + t * (p1 - p0), G.slerp(q0, q1, t), vel[side])

            self._physics_step(control)
        # Make positions/contacts consistent with the current state for observers (and with set_state()).
        mujoco.mj_forward(self.model, self.data)
        self.targets = goal
        self.time += self.dt
        self.step_count += 1
        self._update_lights()
        return self.observe()

    def _update_lights(self):
        """Flicker (OOD): fluorescent-style intensity dips. Rendering only, no effect on physics."""
        fl = self.scenario.flicker if self.scenario else None
        if fl:
            amp, f, ph = fl
            dip = (0.5 + 0.5 * np.sin(2 * np.pi * f * self.time + ph)) ** 6
            self.model.light_diffuse[:] = self._light_base * (1 - amp * dip)
            self.model.vis.headlight.ambient[:] = self._head_base[0] * (1 - amp * dip)
            self.model.vis.headlight.diffuse[:] = self._head_base[1] * (1 - amp * dip)

    def set_compliance(self, side, scale=None, center=None):
        """Per-axis gains (TCP frame x,y,z, rx,ry,rz) on the IK error; <1 makes that axis softer, like
        a low-stiffness impedance controller. `center` (TCP frame) moves the point the gains act on,
        e.g. to the plug tip for remote-centre compliance. None restores full stiffness at the TCP."""
        self.compliance[side] = np.ones(6) if scale is None else np.asarray(scale, float)
        self.compliance_center[side] = None if center is None else np.asarray(center, float)

    def apply_force(self, body: str | int, force, torque=(0, 0, 0), duration_s=0.1):
        """External wrench (world frame) on a body for a duration, e.g. a bump during insertion."""
        b = self.model.body(body).id if isinstance(body, str) else body
        self.ext_force = (b, np.concatenate([force, torque]), max(1, int(duration_s / self.cfg.timestep)))

    # ------------------------------------------------------------------ latch / privileged state
    def _plug_geometry(self):
        """Nearest physical port to the plug tip, with depth / lateral error in the port frame."""
        d = self.data
        tip = d.site_xpos[self.plug_tip_site]
        ax = d.xmat[self.plug_body].reshape(3, 3)[:, 0]
        R = d.xmat[self.rack_body].reshape(3, 3)  # all ports share the rack orientation (+x = insertion)
        rel = (tip - d.site_xpos[self._active_sites]) @ R
        i = int(np.argmin(rel[:, 1] ** 2 + rel[:, 2] ** 2 + 0.01 * np.minimum(rel[:, 0], 0) ** 2))
        depth = float(rel[i, 0])
        lateral = float(np.linalg.norm(rel[i, 1:]))
        angle = float(np.arccos(np.clip(ax @ R[:, 0], -1, 1)))
        return int(self.active[i]), depth, lateral, angle

    def _latch_detent(self):
        pc = self.cfg.port
        if self.latched_port >= 0 or pc.latch_force <= 0:
            return
        k, depth, lateral, _ = self._plug_geometry()
        if depth > pc.latch_engage and lateral < 0.002:
            f = pc.latch_force * min(1.0, (depth - pc.latch_engage) / (pc.seat_depth - pc.latch_engage))
            self.data.xfrc_applied[self.plug_body, :3] -= f * self.data.xmat[self.rack_body].reshape(3, 3)[:, 0]

    def _check_latch(self):
        if self.latched_port >= 0:
            return
        pc = self.cfg.port
        k, depth, lateral, angle = self._plug_geometry()
        if depth >= pc.seat_depth and lateral <= pc.latch_max_lateral and angle <= pc.latch_max_angle:
            d, m = self.data, self.model
            # Weld relpose = plug pose in the latch_anchor (identity at the origin) frame.
            m.eq_data[self.latch_eq, 3:6] = d.xpos[self.plug_body]
            m.eq_data[self.latch_eq, 6:10] = d.xquat[self.plug_body]
            d.eq_active[self.latch_eq] = 1
            self.latched_port = k
            self.latch_time = self.time

    def grasped_by(self):
        """Arm whose two finger pads both touch the plug, or None."""
        d = self.data
        touching = {s: set() for s in ARMS}
        for c in d.contact[: d.ncon]:
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.plug_geoms:
                    for s in ARMS:
                        if b in self.arm[s]["pads"]:
                            touching[s].add(b)
        for s in ARMS:
            if len(touching[s]) == 2:
                return s
        return None

    def privileged(self):
        d = self.data
        k, depth, lateral, angle = self._plug_geometry()
        return dict(
            plug_pos=d.xpos[self.plug_body].copy(),
            plug_quat=d.xquat[self.plug_body].copy(),
            tip_pos=d.site_xpos[self.plug_tip_site].copy(),
            grasp_pos=d.site_xpos[self.plug_grasp_site].copy(),
            nearest_port=k,
            depth=depth,
            lateral=lateral,
            angle=angle,
            latched_port=self.latched_port,
            grasped_by=self.grasped_by(),
            target_port=self.scenario.port_idx,
        )

    def port_pose(self, k):
        s = self.port_sites[k]
        return self.data.site_xpos[s].copy(), G.mat_to_quat(self.data.site_xmat[s])

    def free_ports(self):
        occ = {int(e[0]) for e in self.scenario.occupied}
        return [int(k) for k in self.active if k not in occ]

    # ------------------------------------------------------------------ observations
    def observe(self):
        d = self.data
        parts = []
        for side in ARMS:
            a = self.arm[side]
            p, q = self._tcp_pose(side)
            parts += [d.qpos[a["qadr"]], d.qvel[a["dadr"]], p, q, [d.qpos[a["fing"][0]] + d.qpos[a["fing"][1]]]]
        return np.concatenate(parts).astype(np.float32)

    def _get_renderer(self, w, h):
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (w, h):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
        return self._renderer

    def render(self, camera: str, width=None, height=None, fx=True):
        """Robot camera image, including the scenario's sensor effects (noise, blur, exposure, ...)."""
        r = self._get_renderer(width or self.cfg.camera.width, height or self.cfg.camera.height)
        r.update_scene(self.data, camera=camera)
        img = r.render().copy()
        if fx and self.scenario is not None and self.scenario.image_fx:
            seed = zlib.crc32(f"{self.scenario.seed}/{camera}/{self.step_count}".encode())
            img = apply_image_fx(img, self.scenario.image_fx, seed)
        return img

    def render_free(self, lookat=(0.30, 0, 0.95), distance=1.35, azimuth=90, elevation=-22, width=640, height=400):
        """Render from a free (third-person) camera, for figures (no sensor effects)."""
        r = self._get_renderer(width, height)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    # ------------------------------------------------------------------ save / restore
    STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def get_state(self):
        """Full snapshot: physics + controller state (targets, grip, compliance) + latch + external
        force. Restorable with set_state(); restoring then stepping is deterministic."""
        m, d = self.model, self.data
        phys = np.zeros(mujoco.mj_stateSize(m, self.STATE_SPEC))
        mujoco.mj_getState(m, d, phys, self.STATE_SPEC)
        aux = np.concatenate(
            [
                np.concatenate([self.targets[s][0], self.targets[s][1], [self.grip_cmd[s]], self.compliance[s],
                                np.full(3, np.nan) if self.compliance_center[s] is None else self.compliance_center[s]])
                for s in ARMS
            ]
            + [[self.latched_port, -1 if self.latch_time is None else self.latch_time, self.time, self.step_count], m.eq_data[self.latch_eq]]
            + [[-1] + [0] * 7 if self.ext_force is None else [self.ext_force[0], *self.ext_force[1], self.ext_force[2]]]
        )
        return dict(physics=phys, aux=aux)

    def set_state(self, st):
        m, d = self.model, self.data
        mujoco.mj_setState(m, d, np.asarray(st["physics"], float), self.STATE_SPEC)
        aux = np.asarray(st["aux"], float)
        for i, s in enumerate(ARMS):
            a = aux[17 * i : 17 * i + 17]
            self.targets[s] = (a[:3].copy(), a[3:7].copy())
            self.grip_cmd[s] = float(a[7])
            self.compliance[s] = a[8:14].copy()
            self.compliance_center[s] = None if np.isnan(a[14]) else a[14:17].copy()
        o = 17 * len(ARMS)
        self.latched_port = int(aux[o])
        self.latch_time = None if aux[o + 1] < 0 else float(aux[o + 1])
        self.time = float(aux[o + 2])
        self.step_count = int(aux[o + 3])
        ne = m.eq_data.shape[1]
        m.eq_data[self.latch_eq] = aux[o + 4 : o + 4 + ne]
        ef = aux[o + 4 + ne : o + 4 + ne + 8]
        self.ext_force = None if ef[0] < 0 else (int(ef[0]), ef[1:7].copy(), int(ef[7]))
        mujoco.mj_forward(m, d)
        self._update_lights()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _frame_z(z):
    """Rotation matrix whose z axis is `z` (for orienting capsules)."""
    z = np.asarray(z, float)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)
