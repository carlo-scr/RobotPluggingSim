"""Scripted expert (privileged state machine), standing in for the human teleoperator.

The arm closer to the plug picks it up (top-down grasp on the boot), re-orients it so it points into
the panel, re-measures the plug-in-hand pose (closed loop, as an operator would by looking), aligns
in front of the lit port, and inserts compliantly until the latch clicks. If the plug jams on the jack
lip, it backs off and retries with a small spiral search offset, as an operator would. Then it
releases, retreats and goes home. The other arm stays at home.

Phases: approach -> descend -> close -> lift -> transport -> align -> insert -> hold -> release
        -> retreat -> home -> done                (insert/hold -> backoff -> insert on a jam)
"""
from __future__ import annotations

import numpy as np

from . import geometry as G
from .env import ACT_DIM, ARMS, Q_DOWN, PlugEnv
from .failures import FailureSpec

PHASES = ("approach", "descend", "close", "lift", "transport", "align", "insert", "hold", "release", "retreat", "home", "done")
NEXT = {a: b for a, b in zip(PHASES[:-1], PHASES[1:])}
NEXT["backoff"] = "insert"
# Spiral search offsets (unit steps in the port's y/z plane) tried after successive jams.
SEARCH = np.array([[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]], float)


def down_quat(yaw):
    return G.quat_mul(G.yaw_quat(yaw), Q_DOWN)


class ScriptedExpert:
    PRE_DIST = 0.020  # tip distance in front of the panel face before inserting
    # Phases in which the expert requests the controller's compliant insertion mode (action dim 8).
    COMPLIANT_PHASES = ("insert", "hold", "backoff")
    OVERSHOOT = 0.003  # push this far past the latch depth
    MAX_LEAD = 0.003  # axial target lead over the TCP near the jack (bounds the push force)
    MAX_RETRIES = 3
    SEARCH_STEP = 0.0006
    JAM_WINDOW = 6  # control steps without depth progress that count as a jam

    def __init__(self, env: PlugEnv, failure: FailureSpec | None = None, seed: int = 0, noise: bool = True):
        self.env = env
        self.failure = failure
        self.rng = np.random.default_rng(seed + 7919)
        self.noise = noise
        pr = env.privileged()
        self.arm = "right" if pr["grasp_pos"][1] < 0 else "left"
        self.port = env.scenario.port_idx
        if self._f("wrong_port"):
            self.port = failure.params["port"]
        # A misperceiving operator does not search; it keeps pushing where it believes the port is.
        self.max_retries = 0 if self._f("misalign") or self._f("partial_insert") else self.MAX_RETRIES
        r = self.rng
        self.speed = float(r.uniform(0.85, 1.2)) if noise else 1.0
        self.grasp_jitter = float(r.normal(0, 0.0015)) if noise else 0.0
        self.approach_h = float(r.uniform(0.05, 0.09)) if noise else 0.07
        self.port_noise_yz = r.normal(0, 0.0003, 2) if noise else np.zeros(2)
        self.port_noise_rot = r.normal(0, 0.005, 3) if noise else np.zeros(3)
        self.home_pose = {s: env.targets[s] for s in ARMS}
        self.grip = 0.0
        self.insert_axis = np.array([1.0, 0, 0])
        self.servo_p = np.zeros(3)
        self.servo_r = np.zeros(3)
        self.search = np.zeros(2)
        self.retries = 0
        self.depth_hist = []
        self.events = []  # (name, time)
        self.tp, self.tq = env.targets[self.arm]
        self.phase = None
        self._start("approach")

    # ------------------------------------------------------------------ helpers
    def _f(self, kind):
        return self.failure is not None and self.failure.kind == kind

    @property
    def done(self):
        return self.phase == "done"

    def _plug_in_hand(self):
        """Pose of the TCP in the plug frame (measured, privileged)."""
        pr = self.env.privileged()
        p_t, q_t = self.env._tcp_pose(self.arm)
        pi, qi = G.pose_inv(pr["plug_pos"], pr["plug_quat"])
        return G.pose_mul(pi, qi, p_t, q_t)

    def _believed_port(self):
        """Port pose as the operator perceives it: true pose + perception noise (+ spiral search
        offset, + injected bias for `misalign`), expressed in the port's own frame."""
        p_port, q_port = self.env.port_pose(self.port)
        R = G.quat_to_mat(q_port)
        yz = self.port_noise_yz + self.search
        q_err = G.quat_from_rotvec(self.port_noise_rot)
        if self._f("misalign"):
            yz = yz + np.asarray(self.failure.params["dyz"])
            q_err = G.quat_mul(G.quat_from_rotvec(self.failure.params["drot"]), q_err)
        return p_port + R @ np.array([0.0, yz[0], yz[1]]), G.quat_mul(q_port, q_err)

    def _pre_insert_goal(self):
        pl = self.env.cfg.plug
        p_port, q_des = self._believed_port()
        axis = G.quat_rotate(q_des, [1, 0, 0])
        p_plug = p_port - axis * (self.PRE_DIST + pl.boot_len + pl.body_len)
        rel_p, rel_q = self._plug_in_hand()
        self.insert_axis = axis
        return G.pose_mul(p_plug, q_des, rel_p, rel_q)

    def _start(self, ph):
        env, cfg = self.env, self.env.cfg
        if self.phase in ("insert", "hold"):
            self.servo_p, self.servo_r = np.zeros(3), np.zeros(3)  # already folded into tp/tq
        self.phase = ph
        p0, q0 = self.tp, self.tq
        goal, T = (p0, q0), 0.3
        pr = env.privileged()
        if ph == "approach":
            yaw = G.yaw_of_quat(pr["plug_quat"])
            off = self.grasp_jitter + (self.failure.params["offset"] if self._f("misgrasp") else 0.0)
            ax = np.array([np.cos(yaw), np.sin(yaw), 0])
            self.grasp_p = pr["grasp_pos"] + off * ax + np.array([0, 0, 0.0035])
            goal = (self.grasp_p + [0, 0, self.approach_h], down_quat(yaw))
            T = max(0.9, np.linalg.norm(goal[0] - p0) / 0.25)
        elif ph == "descend":
            goal, T = (self.grasp_p, q0), 0.6
        elif ph == "close":
            self.grip = self.failure.params["grip_cmd"] if self._f("weak_grip") else 1.0
            T = 0.45
        elif ph == "lift":
            goal, T = (p0 + [0, 0, 0.09], q0), 0.6
        elif ph == "transport":
            goal = self._pre_insert_goal()
            rot = np.linalg.norm(G.quat_error(goal[1], q0))
            T = max(1.2, np.linalg.norm(goal[0] - p0) / 0.22, 0.6 + rot / 1.2)
        elif ph == "align":
            goal, T = self._pre_insert_goal(), 0.7
        elif ph == "insert":
            frac = self.failure.params["depth_frac"] if self._f("partial_insert") else 1.0
            extra = 0.0 if self._f("partial_insert") else self.OVERSHOOT
            to_go = cfg.port.seat_depth * frac + extra - pr["depth"]
            goal = (p0 + self.insert_axis * to_go, q0)
            T = 1.4 * max(0.4, to_go / (self.PRE_DIST + cfg.port.seat_depth))
            self.depth_hist = []
        elif ph == "hold":
            T = 0.3
        elif ph == "backoff":
            # Pull the tip back ~4 mm in front of the jack and try a new search offset.
            self.retries += 1
            self.search = self.SEARCH_STEP * np.ceil(self.retries / 8) * SEARCH[(self.retries - 1) % 8]
            goal, T = (p0 - self.insert_axis * (pr["depth"] + 0.004 + self.MAX_LEAD), q0), 0.4
        elif ph == "release":
            self.grip, T = 0.0, 0.4
        elif ph == "retreat":
            goal, T = (p0 - self.insert_axis * 0.05 + [0, 0, 0.03], q0), 0.7
        elif ph == "home":
            goal = self.home_pose[self.arm]
            rot = np.linalg.norm(G.quat_error(goal[1], q0))
            T = max(1.0, np.linalg.norm(goal[0] - p0) / 0.25, 0.5 + rot / 1.2)
        elif ph == "done":
            T = 1e9
        if ph not in ("close", "hold", "release", "backoff", "done"):
            T /= self.speed
        self.seg = dict(p0=p0, q0=q0, p1=np.asarray(goal[0], float), q1=np.asarray(goal[1], float), n=max(1, int(round(T / env.dt))), i=0)
        self.events.append((ph, round(env.time, 4)))

    def _servo(self, gain=1.0):
        """Closed-loop correction while the tip is still in free space: steer the plug tip onto the
        (believed) port axis and the plug axis onto the (believed) insertion direction. (Gain 1 per
        step on the *target*; the compliant controller only applies ~10 % of it laterally.)"""
        pr = self.env.privileged()
        p_port, _ = self._believed_port()
        if (pr["tip_pos"] - p_port) @ self.insert_axis > -0.003:
            return  # at the chamfer: just push, compliantly
        e = p_port - pr["tip_pos"]
        e -= (e @ self.insert_axis) * self.insert_axis
        self.servo_p = np.clip(self.servo_p + gain * e, -0.004, 0.004)
        plug_ax = G.quat_rotate(pr["plug_quat"], [1, 0, 0])
        self.servo_r = np.clip(self.servo_r + 0.3 * gain * np.cross(plug_ax, self.insert_axis), -0.05, 0.05)

    def _jammed(self):
        """No depth progress for JAM_WINDOW steps while holding the plug at the jack, not yet latched.
        (A dropped plug is not a jam: the expert then carries on blindly, like an operator who missed it.)"""
        pr = self.env.privileged()
        if pr["latched_port"] >= 0 or pr["depth"] < -0.003 or pr["grasped_by"] != self.arm or pr["lateral"] > 0.004:
            self.depth_hist = []
            return False
        self.depth_hist.append(pr["depth"])
        h = self.depth_hist[-self.JAM_WINDOW :]
        return len(h) == self.JAM_WINDOW and max(h) - h[0] < 0.0002

    # ------------------------------------------------------------------ policy
    def act(self):
        env, s = self.env, self.seg
        s["i"] += 1
        u = G.min_jerk(s["i"] / s["n"])
        if self.phase in ("insert", "backoff"):
            self._servo()
        self.tp = s["p0"] + u * (s["p1"] - s["p0"]) + self.servo_p
        if self.phase in ("insert", "hold") and env.privileged()["depth"] > -0.004:
            # Force-limited push near the jack: the target may lead the TCP by at most MAX_LEAD.
            p_tcp, _ = env._tcp_pose(self.arm)
            lead = (self.tp - p_tcp) @ self.insert_axis
            if lead > self.MAX_LEAD:
                self.tp = self.tp - (lead - self.MAX_LEAD) * self.insert_axis
        self.tq = G.quat_mul(G.quat_from_rotvec(self.servo_r), G.slerp(s["q0"], s["q1"], u))
        grip = self.grip
        if self._f("drop") and self.phase == "transport":
            k0 = self.failure.params["at_frac"] * s["n"]
            if k0 <= s["i"] < k0 + self.failure.params["open_s"] / env.dt:
                grip = 0.0
        if self._f("bump") and self.phase == self.failure.params["phase"] and s["i"] == max(1, int(self.failure.params["at_frac"] * s["n"])):
            env.apply_force(f"{self.arm[0]}_hand", self.failure.params["force"], self.failure.params["torque"], duration_s=self.failure.params["duration_s"])
            self.events.append(("bump", round(env.time, 4)))
        a = np.zeros((len(ARMS), ACT_DIM))
        i = ARMS.index(self.arm)
        p_cur, q_cur = env.targets[self.arm]
        a[i, :3] = self.tp - p_cur
        a[i, 3:6] = G.quat_error(self.tq, q_cur)
        a[i, 6] = grip
        a[i, 7] = float(self.phase in self.COMPLIANT_PHASES)
        # ---- transitions ----
        if self.phase in ("insert", "hold") and self.retries < self.max_retries and self._jammed():
            self.events.append(("jam", round(env.time, 4)))
            self._start("backoff")
        elif s["i"] >= s["n"] and not self.done:
            nxt = NEXT[self.phase]
            if (self.phase == "hold" and env.latched_port < 0 and self.retries < self.max_retries and not self._f("partial_insert")
                    and env.privileged()["grasped_by"] == self.arm):
                self.events.append(("jam", round(env.time, 4)))
                nxt = "backoff"
            self._start(nxt)
        return a.reshape(-1)
