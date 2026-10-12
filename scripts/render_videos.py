#!/usr/bin/env python3
"""Render showcase videos (mp4) and contact sheets (png) of the plugging sim.

Each frame is a 2x2 mosaic: third-person view | close-up of the target port
                             active gripper front cam | active gripper rear cam
with an overlay of time, phase, insertion depth and latch state.

    python scripts/render_videos.py --out media            # all clips
    python scripts/render_videos.py --out media --clips nominal,drop
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

from plugsim.datagen import EpisodeSpec, build_episode
from plugsim.env import PlugEnv

# name -> (spec, caption)
CLIPS = {
    "nominal": (EpisodeSpec(seed=1003), "nominal: pick, re-orient, align, compliant insert, latch"),
    "nominal_left": (EpisodeSpec(seed=1001), "nominal (left arm)"),
    "jam_retry": (EpisodeSpec(seed=21), "nominal with a jam: back off + spiral retry"),
    "drop": (EpisodeSpec(seed=2003, category="known_failure", failure="drop", severity=0.6), "known failure: cable dropped in transport"),
    "misalign": (EpisodeSpec(seed=2001, category="known_failure", failure="misalign", severity=0.8), "known failure: misaligned, jams on the jack"),
    "partial_insert": (EpisodeSpec(seed=2002, category="known_failure", failure="partial_insert", severity=0.2), "known failure: released before the click, pops out"),
    "wrong_port": (EpisodeSpec(seed=2000, category="known_failure", failure="wrong_port", severity=0.5), "known failure: wrong port"),
    "ood_lighting_distractors": (EpisodeSpec(seed=3000, category="ood", ood=["lighting_dim", "clutter"]), "OOD: dim lighting + clutter"),
    "ood_dense_ports": (EpisodeSpec(seed=3001, category="ood", ood=["dense_ports"]), "OOD: densely occupied panel"),
    "ood_colored_light": (EpisodeSpec(seed=3002, category="ood", ood=["lighting_colored"]), "OOD: coloured lighting"),
    "ood_sensor": (EpisodeSpec(seed=3003, category="ood", ood=["camera_noise", "motion_blur"]), "OOD: sensor noise + motion blur (robot cams)"),
    "ood_rack_offset": (EpisodeSpec(seed=3004, category="ood", ood=["rack_offset"]), "OOD: rack offset/yawed (robot parked differently)"),
}

W, H = 480, 270


def _font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


FONT, FONT_S = _font(17), _font(13)


def label(img, text, xy=(8, 6), font=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = font or FONT_S
    x, y = xy
    bbox = d.textbbox((x, y), text, font=font)
    d.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def frame(env, ex, caption):
    port = env.port_pose(env.scenario.port_idx)[0]
    pr = env.privileged()
    # MuJoCo free camera sits at lookat - distance * (cos el cos az, cos el sin az, sin el).
    sgn = 1.0 if ex.arm == "right" else -1.0  # wide side view from the idle arm's side ...
    wide = env.render_free(lookat=(0.40, 0.0, 0.92), distance=1.5, azimuth=-sgn * 90, elevation=-30, width=W, height=H)
    # ... close-up from in front of the panel, on the other side so the arm doesn't block it.
    close = env.render_free(lookat=port + np.array([-0.025, 0, 0]), distance=0.2, azimuth=-sgn * 35, elevation=-14, width=W, height=H)
    gf = env.render(f"{ex.arm}_gripper_front", width=W, height=H)
    gr = env.render(f"{ex.arm}_gripper_rear", width=W, height=H)
    wide = label(wide, "third person")
    close = label(close, f"target port {env.scenario.port_idx} (lit)")
    gf = label(gf, f"{ex.arm} gripper front cam")
    gr = label(gr, f"{ex.arm} gripper rear cam")
    img = np.concatenate([np.concatenate([wide, close], 1), np.concatenate([gf, gr], 1)], 0)
    latched = "LATCHED" if env.latched_port >= 0 else ""
    status = f"t={env.time:4.1f}s  phase: {ex.phase:<9}  depth {pr['depth'] * 1e3:6.1f} mm  {latched}"
    img = label(img, caption, xy=(10, 2 * H - 52), font=FONT)
    img = label(img, status, xy=(10, 2 * H - 26), font=FONT)
    return img


def render_clip(env, name, spec, caption, out: Path, tail_s=0.8):
    _, _, ex = build_episode(env, spec)
    frames = [frame(env, ex, caption)]
    tail = int(tail_s / env.dt)
    while env.time < env.cfg.episode_max_s and tail > 0 and not env.unstable:
        env.step(ex.act())
        frames.append(frame(env, ex, caption))
        if ex.done:
            tail -= 1
    path = out / f"{name}.mp4"
    w = imageio.get_writer(path, fps=round(1 / env.dt), codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    sheet = np.concatenate([np.concatenate([frames[i][::2, ::2] for i in idx[:3]], 1), np.concatenate([frames[i][::2, ::2] for i in idx[3:]], 1)], 0)
    imageio.imwrite(out / f"{name}.png", sheet)
    ok = env.latched_port == env.scenario.port_idx
    return path, len(frames), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media")
    ap.add_argument("--clips", default="all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PlugEnv()
    names = list(CLIPS) if a.clips == "all" else a.clips.split(",")
    for n in names:
        spec, cap = CLIPS[n]
        path, nf, ok = render_clip(env, n, spec, cap, out)
        print(f"{path}  ({nf} frames, {nf * env.dt:.1f}s, {'success' if ok else 'failure'})", flush=True)


if __name__ == "__main__":
    main()
