"""Notebook helpers: image grids, video embedding, cable plots, and branching rollouts from snapshots."""
from __future__ import annotations

import base64
import copy
import io

import numpy as np

from .env import PlugEnv
from .expert import ScriptedExpert


# ---------------------------------------------------------------------- display
def image_grid(images, titles=None, cols=4, scale=3.2):
    import matplotlib.pyplot as plt

    n = len(images)
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale * h / w + 0.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    return fig


def video_html(frames, fps=20, width=720):
    """Inline mp4 (base64) for notebooks."""
    import imageio.v2 as imageio
    from IPython.display import HTML

    buf = io.BytesIO()
    w = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    b64 = base64.b64encode(buf.getvalue()).decode()
    return HTML(f'<video width="{width}" controls loop autoplay muted><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')


def plot_cable_layouts(cfg, scenarios, ax=None):
    import matplotlib.pyplot as plt

    from .scenario import cable_points

    ax = ax or plt.gca()
    s = cfg.shelf
    ax.add_patch(plt.Rectangle((s.x_min, -s.y_half), s.x_max - s.x_min, 2 * s.y_half, color="0.9", zorder=0))
    ax.plot([cfg.rack.face_x] * 2, [-cfg.rack.panel_half_width, cfg.rack.panel_half_width], "k", lw=4, label="patch panel")
    for sy in (-1, 1):
        ax.plot(cfg.arm.base_x, sy * cfg.arm.base_abs_y, "s", color="0.4", ms=12)
    tip_len = cfg.plug.boot_len + cfg.plug.body_len
    for sc in scenarios:
        anchor = np.array([sc.anchor_x, sc.anchor_side * cfg.cable.anchor_abs_y])
        pts, h = cable_points(cfg, anchor, sc.theta0, np.asarray(sc.deltas), sc.plug_delta)
        tip = pts[-1] + tip_len * np.array([np.cos(h), np.sin(h)])
        (ln,) = ax.plot(pts[:, 0], pts[:, 1], lw=1.5, color=sc.cable_color[:3])
        ax.plot([pts[-1, 0], tip[0]], [pts[-1, 1], tip[1]], lw=4, color="0.2")
        ax.plot(*anchor, "o", color="k", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (towards rack)")
    ax.set_ylabel("y [m]")
    return ax


# ---------------------------------------------------------------------- branching
def clone_expert(ex: ScriptedExpert) -> dict:
    """Snapshot of the expert's internal state (everything except the env reference)."""
    return copy.deepcopy({k: v for k, v in ex.__dict__.items() if k != "env"})


def restore_expert(ex: ScriptedExpert, state: dict):
    ex.__dict__.update(copy.deepcopy(state))
    return ex


def run_until(env: PlugEnv, ex: ScriptedExpert, phase: str, max_s=None):
    """Step the expert until it *enters* `phase` (snapshot point for branching)."""
    max_s = max_s or env.cfg.episode_max_s
    while ex.phase != phase and not ex.done and env.time < max_s:
        env.step(ex.act())
    return ex.phase == phase


def finish(env: PlugEnv, ex: ScriptedExpert, until=("release",), max_s=None, callback=None):
    """Run the expert until one of `until` phases (or done/time-out); optional per-step callback."""
    max_s = max_s or env.cfg.episode_max_s
    while ex.phase not in until and not ex.done and env.time < max_s and not env.unstable:
        a = ex.act()
        env.step(a)
        if callback:
            callback(env, ex, a)
    return env.latched_port == env.scenario.port_idx


def plug_port_force(env: PlugEnv):
    """Total contact force exerted *on the plug* by the panel/jack geoms (world frame, N)."""
    import mujoco

    m, d = env.model, env.data
    panel = set(env.panel_geoms)
    f = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 in env.plug_geoms and g2 in panel:
            sgn = -1.0  # contact force acts on geom2 along +normal; plug is geom1
        elif g2 in env.plug_geoms and g1 in panel:
            sgn = 1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        f += sgn * (c.frame.reshape(3, 3).T @ f6[:3])
    return f


def expert_at(env: PlugEnv, spec, step: int):
    """Rebuild an episode deterministically up to `step` (env + expert internal state), e.g. to branch
    from a recorded episode. Returns the expert; `env` is left in the state at `step`."""
    from .datagen import build_episode

    _, _, ex = build_episode(env, spec)
    for _ in range(step):
        env.step(ex.act())
    return ex


def chunk_then_expert(env: PlugEnv, ex: ScriptedExpert, chunk, until=("release",), max_s=None):
    """Execute an action chunk open-loop (the expert's schedule keeps running underneath, its actions
    are overridden), then hand control back to the expert. Returns success (target port latched)."""
    for a in chunk:
        ex.act()
        env.step(a)
        if ex.phase in until or ex.done:
            break
    return finish(env, ex, until=until, max_s=max_s)


def run_live(cmd, cwd=None, skip=("WARNING",)):
    """Run a script and stream its output into the notebook (raise with the tail of the log on failure)."""
    import subprocess

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
    tail = []
    for line in p.stdout:
        if any(k in line for k in skip):
            continue
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    rc = p.wait()
    if rc:
        raise RuntimeError(f"command failed (exit {rc}); last lines:\n" + "".join(tail))
