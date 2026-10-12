"""Episode rollout, labelling and export.

Each episode is written as one folder in the same layout as the real WR1 data that
GaugeFailClassification's `Trajectory` loader expects:

    ep_000123/
      episode.parquet      timestamp (int64 ns), observation.state, action, + action.ee_target and
                           privileged ground-truth columns (priv.*)
      <camera>.mp4         one video per recorded camera (control rate fps)
      annotations.json     labels: category, outcome, failure spec, OOD shifts, events, scenario
      feature_report.json  names of every state/action dimension, cameras, fps, full EnvCfg
      sim_states.npz       full simulator snapshot per step -> exact counterfactual restarts
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import EnvCfg
from .env import ARMS, PlugEnv, action_names, state_names
from .expert import ScriptedExpert
from .failures import FAILURE_TYPES, FailureSpec, sample_failure
from .scenario import OOD_TYPES, Scenario, sample_scenario

SIM_VERSION = "0.2.0"
DART_PHASES = ("approach", "descend", "lift", "transport", "align")  # free-space phases only


@dataclass
class EpisodeSpec:
    seed: int
    category: str = "nominal"  # nominal | known_failure | ood
    failure: str | None = None
    severity: float = 0.0
    ood: list = field(default_factory=list)
    port_idx: int | None = None
    dart: float = 0.0  # DART noise (m per step on the active arm's translation; rotation 4x in rad)

    def to_dict(self):
        return dict(seed=self.seed, category=self.category, failure=self.failure, severity=self.severity, ood=list(self.ood),
                    port_idx=self.port_idx, dart=self.dart)


def make_plan(n: int, seed0: int = 0, mix=(("nominal", 0.7), ("known_failure", 0.2), ("ood", 0.1)), failure_types=FAILURE_TYPES, ood_types=OOD_TYPES):
    """A list of EpisodeSpecs with the requested category mix. Failure types and OOD shifts are
    drawn uniformly; severities uniformly in [0, 1]; OOD episodes get one or two shifts."""
    rng = np.random.default_rng(seed0)
    cats, probs = zip(*mix)
    plan = []
    for i in range(n):
        cat = str(rng.choice(cats, p=np.asarray(probs) / np.sum(probs)))
        sp = EpisodeSpec(seed=seed0 + i, category=cat)
        if cat == "known_failure":
            sp.failure = str(rng.choice(failure_types))
            sp.severity = float(rng.uniform(0, 1))
        elif cat == "ood":
            sp.ood = [str(x) for x in rng.choice(ood_types, size=int(rng.integers(1, 3)), replace=False)]
        plan.append(sp)
    return plan


def build_episode(env: PlugEnv, spec: EpisodeSpec):
    """Reset env for a spec; returns (scenario, failure, expert)."""
    sc = sample_scenario(env.cfg, spec.seed, ood=spec.ood, port_idx=spec.port_idx)
    env.reset(sc)
    failure = None
    if spec.failure:
        failure = sample_failure(spec.failure, spec.severity, np.random.default_rng(spec.seed + 104729), env.cfg, sc)
    expert = ScriptedExpert(env, failure=failure, seed=spec.seed)
    return sc, failure, expert


def _ee_target(env: PlugEnv):
    return np.concatenate([np.concatenate([env.targets[s][0], env.targets[s][1], [env.grip_cmd[s]]]) for s in ARMS]).astype(np.float32)


def run_episode(env: PlugEnv, spec: EpisodeSpec, cameras=None, save_states=True, max_s=None):
    """Roll out the scripted expert for one spec and return everything needed to write the episode."""
    sc, failure, ex = build_episode(env, spec)
    port_p, port_q = env.port_pose(sc.port_idx)
    cameras = list(env.cfg.camera.record if cameras is None else cameras)
    max_s = max_s or env.cfg.episode_max_s
    rec = {k: [] for k in ("t", "state", "action", "action_expert", "ee_target", "phase", "plug_pos", "plug_quat", "tip_pos", "depth", "lateral", "angle", "latched", "grasped")}
    frames = {c: [] for c in cameras}
    phys, aux, act64 = [], [], []
    grasp_t, drop_t, ungrasped_n, was_grasped, max_depth = None, [], 0, False, -1.0
    while not ex.done and env.time < max_s and not env.unstable:
        obs = env.observe()
        for c in cameras:
            frames[c].append(env.render(c))
        pr = env.privileged()
        target_before = _ee_target(env)
        phase = ex.phase
        a_exp = env.clip_action(ex.act())
        a = a_exp
        if spec.dart > 0 and phase in DART_PHASES:
            # DART (Laskey et al. 2017): execute a perturbed action, record the expert's clean
            # (corrective) action as the BC label, so the demos contain recoveries.
            rng_d = np.random.default_rng((spec.seed * 1000003 + env.step_count) % (1 << 32))
            i = 8 * ARMS.index(ex.arm)
            a = a_exp.copy()
            a[i : i + 3] += rng_d.normal(0, spec.dart, 3)
            a[i + 3 : i + 6] += rng_d.normal(0, 4 * spec.dart, 3)
            a = env.clip_action(a)  # record what the env executes
        if save_states:
            # Snapshot *after* the expert acted (it may switch controller modes), right before stepping:
            # restore(step k) + env.step(action k) reproduces the recording exactly.
            st = env.get_state()
            phys.append(st["physics"])
            aux.append(st["aux"])
            act64.append(a)
        rec["t"].append(env.time)
        rec["state"].append(obs)
        rec["action"].append(a.astype(np.float32))
        rec["action_expert"].append(a_exp.astype(np.float32))
        rec["ee_target"].append(target_before)
        rec["phase"].append(phase)
        rec["plug_pos"].append(pr["plug_pos"])
        rec["plug_quat"].append(pr["plug_quat"])
        rec["tip_pos"].append(pr["tip_pos"])
        rec["depth"].append(pr["depth"])
        rec["lateral"].append(pr["lateral"])
        rec["angle"].append(pr["angle"])
        rec["latched"].append(pr["latched_port"])
        g = pr["grasped_by"]
        rec["grasped"].append(-1 if g is None else ARMS.index(g))
        if pr["nearest_port"] == sc.port_idx or pr["lateral"] < 0.01:
            max_depth = max(max_depth, pr["depth"])
        # ---- event detection (privileged) ----
        if g == ex.arm:
            if grasp_t is None:
                grasp_t = env.time
            was_grasped, ungrasped_n = True, 0
        elif was_grasped and pr["latched_port"] < 0 and phase not in ("release", "retreat", "home", "done"):
            ungrasped_n += 1
            if ungrasped_n == 3:
                drop_t.append(round(env.time - 2 * env.dt, 4))
                was_grasped = False
        env.step(a)
    pr = env.privileged()
    latched = env.latched_port
    success = (latched == sc.port_idx) and pr["grasped_by"] is None and not env.unstable
    if success:
        mode = "none"
    elif env.unstable:
        mode = "sim_unstable"
    elif latched >= 0:
        mode = "wrong_port"
    elif grasp_t is None:
        mode = "grasp_missed"
    elif drop_t:
        mode = "dropped"
    elif max_depth > 0.001:
        mode = "not_seated"  # entered the jack but never clicked (incl. popped back out)
    elif any(e[0] in ("insert", "jam") for e in ex.events):
        mode = "jammed"  # reached the jack but never got in
    else:
        mode = "other"
    ann = dict(
        sim_version=SIM_VERSION,
        spec=spec.to_dict(),
        category=spec.category,
        outcome="success" if success else "failure",
        failure_mode=mode,
        injected_failure=None if failure is None else failure.to_dict(),
        ood=list(spec.ood),
        target_port=int(sc.port_idx),
        target_port_pos=[float(x) for x in port_p],
        target_port_quat=[float(x) for x in port_q],
        latched_port=int(latched),
        latch_time_s=env.latch_time,
        active_arm=ex.arm,
        retries=int(ex.retries),
        grasp_time_s=grasp_t,
        drop_times_s=drop_t,
        injection_time_s=_injection_time(ex, failure),
        max_insert_depth=round(float(max_depth), 5),
        phase_events=ex.events,
        duration_s=round(env.time, 4),
        n_steps=len(rec["t"]),
        unstable=bool(env.unstable),
        scenario=sc.to_dict(),
    )
    data = {k: np.asarray(v) for k, v in rec.items()}
    return dict(annotations=ann, data=data, frames=frames, physics=np.asarray(phys), aux=np.asarray(aux), actions64=np.asarray(act64))


def _injection_time(ex: ScriptedExpert, failure: FailureSpec | None):
    """When the injected perturbation starts to act (first time it can influence the rollout)."""
    if failure is None:
        return None
    ev = {}
    for name, t in ex.events:
        ev.setdefault(name, t)
    k = failure.kind
    if k == "bump":
        return ev.get("bump")
    if k == "drop":
        t0 = ev.get("transport")
        t1 = ev.get("align")
        return None if t0 is None or t1 is None else round(t0 + failure.params["at_frac"] * (t1 - t0), 4)
    phase = {"weak_grip": "close", "misgrasp": "approach", "misalign": "transport", "partial_insert": "insert", "wrong_port": "transport"}[k]
    return ev.get(phase)


def write_episode(ep: dict, out_dir: str | Path, fps: float, cfg: EnvCfg):
    import imageio.v2 as imageio
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = ep["data"]
    n = len(d["t"])
    df = pd.DataFrame(
        {
            "timestamp": np.round(d["t"] * 1e9).astype(np.int64),
            "frame_index": np.arange(n),
            "observation.state": list(d["state"]),
            "action": list(d["action"]),
            "action.ee_target": list(d["ee_target"]),
            "action.expert": list(d["action_expert"]),
            "priv.phase": d["phase"],
            "priv.plug_pos": list(d["plug_pos"].astype(np.float32)),
            "priv.plug_quat": list(d["plug_quat"].astype(np.float32)),
            "priv.tip_pos": list(d["tip_pos"].astype(np.float32)),
            "priv.insert_depth": d["depth"].astype(np.float32),
            "priv.lateral_err": d["lateral"].astype(np.float32),
            "priv.angle_err": d["angle"].astype(np.float32),
            "priv.latched_port": d["latched"].astype(np.int16),
            "priv.grasped_by": d["grasped"].astype(np.int8),
        }
    )
    df.to_parquet(out / "episode.parquet", engine="pyarrow")
    for cam, fr in ep["frames"].items():
        if not fr:
            continue
        w = imageio.get_writer(out / f"{cam}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
        for f in fr:
            w.append_data(f)
        w.close()
    if len(ep["physics"]):
        np.savez_compressed(out / "sim_states.npz", physics=ep["physics"], aux=ep["aux"], actions=ep["actions64"])
    with open(out / "annotations.json", "w") as f:
        json.dump(ep["annotations"], f, indent=1, default=_json_default)
    fr = dict(
        sim_version=SIM_VERSION,
        fps=fps,
        observation_state_names=state_names(),
        action_names=action_names(),
        ee_target_names=[f"{s}.{n}" for s in ARMS for n in ("x", "y", "z", "qw", "qx", "qy", "qz", "grip")],
        cameras=list(ep["frames"].keys()),
        image_shape=[cfg.camera.height, cfg.camera.width, 3],
        action_semantics="per arm: TCP delta (world frame, m), rotation-vector delta (world frame, rad), grip in [0,1] (1=close), compliant in {0,1} (insertion impedance mode); deltas are applied to the controller target",
        env_cfg=cfg.to_dict(),
    )
    with open(out / "feature_report.json", "w") as f:
        json.dump(fr, f, indent=1, default=_json_default)
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def summary_row(ann: dict, path: str):
    inj = ann["injected_failure"] or {}
    return dict(
        episode=os.path.basename(path),
        path=path,
        seed=ann["spec"]["seed"],
        category=ann["category"],
        outcome=ann["outcome"],
        failure_mode=ann["failure_mode"],
        injected=inj.get("kind"),
        severity=inj.get("severity"),
        ood="+".join(ann["ood"]),
        target_port=ann["target_port"],
        latched_port=ann["latched_port"],
        active_arm=ann["active_arm"],
        retries=ann["retries"],
        duration_s=ann["duration_s"],
        n_steps=ann["n_steps"],
        unstable=ann["unstable"],
    )


# ---------------------------------------------------------------------- loading / counterfactual restarts
def load_episode(path: str | Path, load_frames=False):
    import pandas as pd

    path = Path(path)
    out = dict(
        df=pd.read_parquet(path / "episode.parquet"),
        annotations=json.load(open(path / "annotations.json")),
        features=json.load(open(path / "feature_report.json")),
    )
    if (path / "sim_states.npz").exists():
        z = np.load(path / "sim_states.npz")
        out["physics"], out["aux"], out["actions64"] = z["physics"], z["aux"], z["actions"]
    if load_frames:
        import imageio.v2 as imageio

        out["frames"] = {c: np.stack(imageio.mimread(path / f"{c}.mp4", memtest=False)) for c in out["features"]["cameras"]}
    return out


def restore(env: PlugEnv, episode: dict, step: int):
    """Put `env` into the exact simulator state of `episode` at `step` (for counterfactual rollouts).
    The env must have been built with the same EnvCfg."""
    env.reset(Scenario.from_dict(episode["annotations"]["scenario"]))
    env.set_state(dict(physics=episode["physics"][step], aux=episode["aux"][step]))
    return env
