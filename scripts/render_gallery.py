#!/usr/bin/env python3
"""Stills and videos of the scene for slides / the README.

    python scripts/render_gallery.py --out media/gallery            # everything
    python scripts/render_gallery.py --out media/gallery --only orbit,ood

Outputs (1280x720 unless noted):
  hero_*.png          third-person views of the aisle mid-task
  panel_*.png         patch-panel close-ups at three occupancy levels
  robot_cameras.png   all seven robot cameras
  lighting.png        nominal + the four lighting OOD shifts
  ood_gallery.png     every OOD type (active gripper cam + head cam)
  orbit.mp4           camera orbit around a full expert episode
  ood_montage.mp4     short clips of visual OOD conditions (head + gripper cam)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plugsim.env import PlugEnv
from plugsim.expert import ScriptedExpert
from plugsim.scenario import OOD_GROUPS, OOD_TYPES, sample_scenario
from plugsim.tools import run_until

W, H = 1280, 720
HERO = {  # name: (lookat, distance, azimuth, elevation)
    "hero_over_shoulder": ((0.38, 0.0, 0.98), 2.0, 60, -25),
    "hero_side_high": ((0.40, 0.0, 0.95), 1.6, 30, -35),
    "hero_left": ((0.40, 0.0, 1.00), 1.8, -45, -20),
    "hero_top": ((0.36, 0.0, 0.90), 1.4, 15, -55),
}


def _font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def label(img, text, size=22, xy=(12, 10)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font(size)
    b = d.textbbox(xy, text, font=f)
    d.rectangle((b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4), fill=(0, 0, 0))
    d.text(xy, text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def grid(tiles, cols):
    rows = [np.concatenate(tiles[i : i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i : i + cols])), 1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, 0)


class Bright:
    """Temporarily boost the lights for third-person beauty shots (robot cameras stay untouched)."""

    def __init__(self, env, k=1.35):
        self.env, self.k = env, k

    def __enter__(self):
        self.saved = self.env.model.light_diffuse.copy()
        self.env.model.light_diffuse[:] *= self.k

    def __exit__(self, *a):
        self.env.model.light_diffuse[:] = self.saved


def mid_task(env, seed, phase="align", extra=8, ood=()):
    env.reset(sample_scenario(env.cfg, seed, ood=list(ood)))
    ex = ScriptedExpert(env, seed=seed)
    run_until(env, ex, phase)
    for _ in range(extra):
        env.step(ex.act())
    return ex


def heroes(env, out):
    mid_task(env, 12)
    with Bright(env):
        for name, (la, d, az, el) in HERO.items():
            imageio.imwrite(out / f"{name}.png", env.render_free(lookat=la, distance=d, azimuth=az, elevation=el, width=W, height=H))
    ex = mid_task(env, 12, "insert", 12)
    port = env.port_pose(env.scenario.port_idx)[0]
    with Bright(env):
        imageio.imwrite(out / "hero_insertion_closeup.png", env.render_free(lookat=port + np.array([-0.03, 0, 0]), distance=0.25, azimuth=-30, elevation=-12, width=W, height=H))


def _seed_with_occupancy(cfg, lo, hi, ood=(), start=0):
    for s in range(start, start + 500):
        if lo <= len(sample_scenario(cfg, s, ood=list(ood)).occupied) <= hi:
            return s
    return start


def panels(env, out):
    c = env.cfg
    for name, (lo, hi), ood in (("panel_sparse", (4, 8), []), ("panel_typical", (14, 20), []), ("panel_dense", (40, 48), ["dense_ports"])):
        seed = _seed_with_occupancy(c, lo, hi, ood, start=20)
        sc = sample_scenario(env.cfg, seed, ood=ood)
        env.reset(sc)
        with Bright(env):
            img = env.render_free(lookat=(env.cfg.rack.face_x - 0.02, 0, env.cfg.rack.row_z - 0.02), distance=0.62, azimuth=-12, elevation=-8, width=W, height=H)
        imageio.imwrite(out / f"{name}.png", label(img, f"{len(sc.occupied)} of {env.cfg.rack.n_ports} ports occupied - target #{sc.port_idx + 1} (blue LED)"))


def cameras(env, out):
    ex = mid_task(env, 12, "align", 4)
    cams = env.cfg.camera.all_cameras
    tiles = [label(env.render(c, 640, 360), c, 18) for c in cams]
    imageio.imwrite(out / "robot_cameras.png", grid(tiles, 4))


def lighting(env, out):
    tiles = []
    for o in ["nominal"] + list(OOD_GROUPS["lighting"]):
        ex = mid_task(env, 15, "align", 4, ood=[] if o == "nominal" else [o])
        if o == "lighting_flicker":  # show a dip
            fl = env.scenario.flicker
            env.time = (np.pi / 2 - fl[2]) / (2 * np.pi * fl[1]) % (1 / fl[1])
            env._update_lights()
        tiles.append(label(env.render("head_front", 640, 360), o, 18))
    imageio.imwrite(out / "lighting.png", grid(tiles, 3))


def ood(env, out):
    tiles = []
    for o in ["nominal"] + list(OOD_TYPES):
        ex = mid_task(env, 7, "insert", 2, ood=[] if o == "nominal" else [o])
        t = np.concatenate([env.render(f"{ex.arm}_gripper_front", 480, 270), env.render("head_front", 480, 270)], 1)
        tiles.append(label(t, o, 18))
    imageio.imwrite(out / "ood_gallery.png", grid(tiles, 3))


def orbit(env, out, seed=12):
    env.reset(sample_scenario(env.cfg, seed))
    ex = ScriptedExpert(env, seed=seed)
    w = imageio.get_writer(out / "orbit.mp4", fps=20, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    n = 0
    sgn = 1.0 if ex.arm == "right" else -1.0  # stay on the active arm's side, never behind the torso
    port = env.port_pose(env.scenario.port_idx)[0]
    t_ins = None
    while not ex.done and env.time < 14:
        env.step(ex.act())
        if ex.phase in ("align", "insert") and t_ins is None:
            t_ins = env.time
        u = min(1.0, env.time / 4.0)  # sweep while picking
        z = 0.0 if t_ins is None else float(np.clip((env.time - t_ins) / 1.5, 0, 1))  # zoom in for the insertion
        if ex.phase in ("retreat", "home", "done"):
            z = max(0.0, 1 - (env.time - (t_ins or 0) - 3.0) / 1.5)
        s_ = z * z * (3 - 2 * z)
        az = sgn * ((85 - 45 * u) * (1 - s_) + 25 * s_)
        el = (-22 - 12 * u) * (1 - s_) - 12 * s_
        d = (2.0 - 0.4 * u) * (1 - s_) + 0.32 * s_
        la = np.array([0.40, 0.0, 0.95]) * (1 - s_) + (port - np.array([0.03, 0, 0])) * s_
        with Bright(env):
            img = env.render_free(lookat=la, distance=d, azimuth=az, elevation=el, width=W, height=H)
        w.append_data(label(img, f"t = {env.time:4.1f} s   phase: {ex.phase}", 22))
        n += 1
    w.close()
    return n


def ood_montage(env, out, secs=3.0):
    shifts = ["lighting_colored", "lighting_harsh", "lighting_flicker", "dense_ports", "clutter", "distractor_cable", "camera_noise", "motion_blur", "overexposure", "lens_occlusion", "panel_color", "cable_color"]
    w = imageio.get_writer(out / "ood_montage.mp4", fps=20, codec="libx264", quality=6, macro_block_size=8, pixelformat="yuv420p")
    for i, o in enumerate(shifts):
        sc = sample_scenario(env.cfg, 400 + i, ood=[o])
        env.reset(sc)
        ex = ScriptedExpert(env, seed=400 + i)
        run_until(env, ex, "transport")
        for _ in range(int(secs / env.dt)):
            env.step(ex.act())
            img = np.concatenate([env.render("head_front", 480, 270), env.render(f"{ex.arm}_gripper_front", 480, 270)], 1)
            w.append_data(label(img, f"OOD: {o}", 18))
    w.close()


# ---------------------------------------------------------------------- more videos
def _writer(path, fps=20):
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")


def _run_expert_frames(env, sc, seed, render, max_s=14.0, failure=None):
    """Run the expert on a scenario and collect render(env, ex) frames each step."""
    env.reset(sc)
    ex = ScriptedExpert(env, failure=failure, seed=seed)
    frames = [render(env, ex)]
    while not ex.done and env.time < max_s:
        env.step(ex.act())
        frames.append(render(env, ex))
    return frames, ex


def tour(env, out, seed=13):
    """Dolly along the rack row, then reveal the robot working."""
    sc = sample_scenario(env.cfg, seed)
    env.reset(sc)
    ex = ScriptedExpert(env, seed=seed)
    w = _writer(out / "tour.mp4")
    fx = env.cfg.rack.face_x
    while not ex.done and env.time < 11:
        env.step(ex.act())
        u = float(np.clip(env.time / 4.5, 0, 1))
        v = float(np.clip((env.time - 4.5) / 3.0, 0, 1))
        e = lambda a: a * a * (3 - 2 * a)
        la = np.array([fx + 0.02, -1.25 + 1.05 * e(u), 1.15]) * (1 - e(v)) + np.array([0.33, 0.0, 0.98]) * e(v)
        az, el, d = (10 + 15 * u) * (1 - e(v)) + 55 * e(v), -8 * (1 - e(v)) - 26 * e(v), 1.25 * (1 - e(v)) + 1.9 * e(v)
        with Bright(env, 1.4):
            img = env.render_free(lookat=la, distance=d, azimuth=az, elevation=el, width=W, height=H)
        w.append_data(img)
    w.close()


def robot_pov(env, out, seed=13):
    """All relevant robot cameras through a full episode (what the policy/WM sees)."""
    sc = sample_scenario(env.cfg, seed)
    w = _writer(out / "robot_pov.mp4")
    def render(env, ex):
        oth = "left" if ex.arm == "right" else "right"
        cams = ["head_front", "chest_left", "chest_right", f"{ex.arm}_gripper_front", f"{ex.arm}_gripper_rear", f"{oth}_gripper_front"]
        tiles = [label(env.render(c, 416, 234), c, 14, (6, 4)) for c in cams]
        return label(grid(tiles, 3), f"t = {env.time:4.1f} s   phase: {ex.phase}", 16, (8, 446))
    frames, _ = _run_expert_frames(env, sc, seed, render)
    for f in frames:
        w.append_data(f)
    w.close()


def failures_montage(env, out):
    """Each known failure type, cut around the moment it shows."""
    from plugsim.datagen import EpisodeSpec, build_episode
    from plugsim.failures import FAILURE_TYPES

    w = _writer(out / "failures_montage.mp4")
    for i, kind in enumerate(FAILURE_TYPES):
        spec = EpisodeSpec(seed=2100 + i, category="known_failure", failure=kind, severity=0.85)
        sc_, fail, ex = build_episode(env, spec)
        frames, times = [], []
        while not ex.done and env.time < 14:
            env.step(ex.act())
            port = env.port_pose(sc_.port_idx)[0]
            a = env.render_free(lookat=(0.40, 0.0, 0.95), distance=1.6, azimuth=(55 if ex.arm == "right" else -55), elevation=-28, width=640, height=360)
            b = env.render(f"{ex.arm}_gripper_front", 640, 360)
            frames.append(label(np.concatenate([a, b], 1), f"failure: {kind}  (severity 0.85)   phase: {ex.phase}   latched: {env.latched_port >= 0}", 18))
            times.append(env.time)
        t_ev = next((t for n, t in ex.events if n in ("bump",)), None)
        t0 = {"drop": ("transport", -0.5), "weak_grip": ("lift", -0.5), "misgrasp": ("descend", 0.0), "misalign": ("align", 0.0),
              "partial_insert": ("insert", 0.5), "wrong_port": ("align", 0.0), "bump": ("align", -0.5)}[kind]
        start = next((t for n, t in ex.events if n == t0[0]), 0.0) + t0[1]
        for f, t in zip(frames, times):
            if start <= t <= start + 4.0:
                w.append_data(f)
    w.close()


def conditions_grid(env, out, seed=16):
    """The same episode under six visual conditions (lighting/sensor shifts do not change physics)."""
    conds = ["nominal", "lighting_dim", "lighting_harsh", "lighting_colored", "camera_noise", "overexposure"]
    clips = []
    for o in conds:
        sc = sample_scenario(env.cfg, seed, ood=[] if o == "nominal" else [o])
        frames, _ = _run_expert_frames(env, sc, seed, lambda e, x, o=o: label(e.render("head_front", 427, 240), o, 14, (6, 4)))
        clips.append(frames)
    n = min(len(c) for c in clips)
    w = _writer(out / "conditions_grid.mp4")
    for t in range(n):
        w.append_data(grid([c[t] for c in clips], 3))
    w.close()


def counterfactual_branches(env, out, seed=None):
    """One saved state, four candidate continuations (growing alignment error): the RQ1 mechanism."""
    from plugsim.tools import clone_expert, finish, restore_expert

    if seed is None:  # sparse panel + top-row target, so nothing hangs in front of the jack
        seed = next(s for s in range(200, 800) if len(sample_scenario(env.cfg, s).occupied) <= 4 and sample_scenario(env.cfg, s).port_idx < env.cfg.rack.cols)
    env.reset(sample_scenario(env.cfg, seed))
    for g in env.cable_geoms + env.boot_geoms:  # make our cable stand out
        env.model.geom_rgba[g] = (1.0, 0.45, 0.05, 1)
    env.model.mat_rgba[env.model.material("boot").id] = (1.0, 0.45, 0.05, 1)
    ex = ScriptedExpert(env, seed=seed, noise=False)
    run_until(env, ex, "align")
    snap, exs = env.get_state(), clone_expert(ex)
    port = env.port_pose(env.scenario.port_idx)[0]
    clips = []
    for off in (0.0, 1.2, 2.2, 3.2):
        env.set_state(snap)
        restore_expert(ex, exs)
        ex.port_noise_yz = np.array([0.0, -off * 1e-3])
        ex.max_retries = 0
        ex._start("align")
        frames = []
        def cb(e, x, a):
            with Bright(e, 1.4):
                img = e.render_free(lookat=port + np.array([-0.025, 0, 0.0]), distance=0.17, azimuth=-25, elevation=-32, width=640, height=360)
            frames.append(img)
        ok = finish(env, ex, until=("retreat",), callback=cb)
        tag = f"branch: belief offset {off:.1f} mm  ->  {'LATCHED' if ok else 'jammed / not seated'}"
        clips.append([label(f, tag, 16, (8, 6)) for f in frames])
    n = max(len(c) for c in clips)
    w = _writer(out / "counterfactual_branches.mp4")
    for t in range(n):
        w.append_data(grid([c[min(t, len(c) - 1)] for c in clips], 2))
    w.close()


def policy_rollouts(env, out, ckpt="runs/policy_state_test.pt", seeds=(50000, 50001)):
    """The learned Diffusion Policy (state obs): a success next to a failure."""
    from plugsim.learn.policy import PolicyRunner, load_policy

    if not Path(ckpt).exists():
        print("no policy checkpoint, skipping")
        return
    pol = load_policy(ckpt)
    clips = []
    for s in seeds:
        runner = PolicyRunner(pol, env, seed=0)
        sc = sample_scenario(env.cfg, s)
        env.reset(sc)
        frames, t_l = [], None
        arm = "right" if env.privileged()["grasp_pos"][1] < 0 else "left"
        while env.time < 12:
            env.step(runner.act())
            a = env.render_free(lookat=(0.40, 0.0, 0.95), distance=1.6, azimuth=(55 if arm == "right" else -55), elevation=-28, width=640, height=360)
            frames.append(label(a, f"learned Diffusion Policy, seed {s}: {'LATCHED' if env.latched_port >= 0 else '...'}", 16))
            if env.latched_port >= 0 and t_l is None:
                t_l = env.time
            if t_l and env.time - t_l > 1.0:
                break
        clips.append(frames)
    n = max(len(c) for c in clips)
    w = _writer(out / "policy_rollouts.mp4")
    for t in range(n):
        w.append_data(np.concatenate([c[min(t, len(c) - 1)] for c in clips], 1))
    w.close()


def dense_insertion_closeup(env, out, seed=None):
    seed = seed if seed is not None else _seed_with_occupancy(env.cfg, 40, 48, ["dense_ports"], start=60)
    sc = sample_scenario(env.cfg, seed, ood=["dense_ports"])
    env.reset(sc)
    ex = ScriptedExpert(env, seed=seed)
    port = env.port_pose(sc.port_idx)[0]
    w = _writer(out / "dense_insertion_closeup.mp4")
    while not ex.done and env.time < 14:
        env.step(ex.act())
        if ex.phase in ("transport", "align", "insert", "hold", "release", "retreat"):
            with Bright(env, 1.3):
                img = env.render_free(lookat=port + np.array([-0.04, 0, -0.005]), distance=0.3, azimuth=-25, elevation=-6, width=W, height=H)
            w.append_data(label(img, f"dense panel ({len(sc.occupied)}/48 occupied)   phase: {ex.phase}   {'LATCHED' if env.latched_port >= 0 else ''}", 20))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media/gallery")
    ap.add_argument("--only", default="heroes,panels,cameras,lighting,ood,orbit,ood_montage,tour,robot_pov,failures_montage,"
                                      "conditions_grid,counterfactual_branches,policy_rollouts,dense_insertion_closeup")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PlugEnv()
    for job in a.only.split(","):
        globals()[job](env, out)
        print("done:", job, flush=True)


if __name__ == "__main__":
    main()
