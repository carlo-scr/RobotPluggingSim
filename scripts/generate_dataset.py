#!/usr/bin/env python3
"""Generate a labelled cable-plugging dataset in parallel (one MuJoCo env per worker process).

Local (Mac):   python scripts/generate_dataset.py --out data/smoke --n 20 --workers 4
H100 node:     MUJOCO_GL=egl python scripts/generate_dataset.py --out /scratch/plugsim/v0 --n 20000 --workers 96

Physics and rendering run on CPU cores (rendering via EGL on the GPU when MUJOCO_GL=egl), so the
throughput scales with the number of cores on the node. Episodes are independent and seeded, so a
run can be split across nodes with --seed0 / --n and the index files concatenated afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ENV = None


def _init(cfg_overrides):
    global _ENV
    from plugsim.config import EnvCfg
    from plugsim.env import PlugEnv

    cfg = EnvCfg()
    for k, v in (cfg_overrides or {}).items():
        obj = cfg
        *path, last = k.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, type(getattr(obj, last))(v) if not isinstance(getattr(obj, last), tuple) else tuple(v))
    _ENV = PlugEnv(cfg)


def _work(args):
    spec_d, out_root, cameras, save_states = args
    from plugsim.datagen import EpisodeSpec, run_episode, summary_row, write_episode

    spec = EpisodeSpec(**spec_d)
    path = Path(out_root) / f"ep_{spec.seed:07d}"
    try:
        ep = run_episode(_ENV, spec, cameras=cameras, save_states=save_states)
        write_episode(ep, path, fps=_ENV.cfg.control_hz, cfg=_ENV.cfg)
        return summary_row(ep["annotations"], str(path)), None
    except Exception:
        return None, (spec_d, traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--mix", default="nominal:0.7,known_failure:0.2,ood:0.1", help="category:weight,...")
    ap.add_argument("--cameras", default=None, help="comma-separated; default = cfg.camera.record; 'none' for physics only")
    ap.add_argument("--no-states", action="store_true", help="skip sim_states.npz (saves disk)")
    ap.add_argument("--dart", type=float, default=0.0, help="DART noise (m/step) for nominal demos (policy training data)")
    ap.add_argument("--cfg", default="{}", help='JSON EnvCfg overrides, e.g. \'{"port.clearance": 0.0006, "camera.width": 224}\'')
    a = ap.parse_args()

    from multiprocessing import get_context

    import pandas as pd

    from plugsim.datagen import make_plan

    mix = tuple((k, float(v)) for k, v in (kv.split(":") for kv in a.mix.split(",")))
    plan = make_plan(a.n, seed0=a.seed0, mix=mix)
    for p in plan:
        p.dart = a.dart if p.category == "nominal" else 0.0
    cams = None if a.cameras is None else ([] if a.cameras == "none" else a.cameras.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"plan_{a.seed0}.json", "w") as f:
        json.dump(dict(args=vars(a), plan=[p.to_dict() for p in plan]), f, indent=1)

    jobs = [(p.to_dict(), str(out), cams, not a.no_states) for p in plan]
    rows, errors, t0 = [], [], time.time()
    ctx = get_context("spawn")  # never share GL contexts across fork
    with ctx.Pool(a.workers, initializer=_init, initargs=(json.loads(a.cfg),)) as pool:
        for i, (row, err) in enumerate(pool.imap_unordered(_work, jobs, chunksize=1)):
            if row:
                rows.append(row)
            if err:
                errors.append(err)
                print("ERROR", err[0], err[1].splitlines()[-1], flush=True)
            if (i + 1) % max(1, a.n // 20) == 0 or i + 1 == a.n:
                el = time.time() - t0
                print(f"[{i + 1}/{a.n}] {el:.0f}s elapsed, {(i + 1) / el:.2f} ep/s, eta {(a.n - i - 1) * el / (i + 1):.0f}s", flush=True)
    idx = pd.DataFrame(rows).sort_values("seed")
    idx.to_parquet(out / f"index_{a.seed0}.parquet")
    if errors:
        with open(out / f"errors_{a.seed0}.json", "w") as f:
            json.dump(errors, f, indent=1)
    print(idx.groupby(["category", "outcome"]).size().to_string())
    print(f"done: {len(rows)} episodes, {len(errors)} errors, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
