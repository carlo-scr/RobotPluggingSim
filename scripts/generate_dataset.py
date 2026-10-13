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
# One BLAS/OpenMP thread per worker: the IK solves tiny 6x6 systems, and multi-threaded BLAS in every
# process oversubscribes the cores (this made Colab generation ~6x slower). Spawned workers inherit it.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
    ap.add_argument("--stall-s", type=float, default=300.0, help="abort if no episode finishes for this long")
    ap.add_argument("--max-render-ms", type=float, default=200.0, help="abort if a frame takes longer (software rendering)")
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

    # Resume: episodes that already have annotations.json are kept (re-running continues where it stopped).
    from plugsim.datagen import summary_row

    rows, todo = [], []
    for p in plan:
        ep = out / f"ep_{p.seed:07d}"
        if (ep / "annotations.json").exists():
            rows.append(summary_row(json.load(open(ep / "annotations.json")), str(ep)))
        else:
            todo.append(p)
    if rows:
        print(f"resuming: {len(rows)} episodes already done, {len(todo)} to go", flush=True)

    # Fail fast if rendering does not work here (instead of workers dying silently).
    if cams is None or cams:
        _init(json.loads(a.cfg))
        cam = (cams or list(_ENV.cfg.camera.record))[0]
        _ENV.render(cam)  # warm-up
        t = time.time()
        for _ in range(5):
            _ENV.render(cam)
        ms = (time.time() - t) / 5 * 1000
        print(f"render check: {ms:.0f} ms/frame (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})", flush=True)
        if ms > a.max_render_ms:
            print(f"ABORTED: rendering is far too slow ({ms:.0f} ms/frame, expected ~2-30 ms): this is CPU (software) rendering.\n"
                  "On Colab: the NVIDIA EGL vendor file is missing - re-run the notebook's first (setup) cell, which writes\n"
                  "/usr/share/glvnd/egl_vendor.d/10_nvidia.json, then retry. Override with --max-render-ms.", flush=True)
            sys.exit(2)

    jobs = [(p.to_dict(), str(out), cams, not a.no_states) for p in todo]
    errors, t0, n = [], time.time(), len(jobs)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    ctx = get_context("spawn")  # never share GL contexts across fork
    # ProcessPoolExecutor (unlike multiprocessing.Pool) raises if a worker process dies instead of hanging.
    ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, initializer=_init, initargs=(json.loads(a.cfg),))
    pending = {ex.submit(_work, j) for j in jobs}
    done_n = 0
    try:
        while pending:
            done, pending = wait(pending, timeout=a.stall_s, return_when=FIRST_COMPLETED)
            if not done:
                raise RuntimeError(f"no episode finished in {a.stall_s:.0f}s: workers look stuck (check rendering / MUJOCO_GL). "
                                   f"Re-run the same command to resume.")
            for f in done:
                row, err = f.result()  # raises BrokenProcessPool if a worker crashed
                done_n += 1
                if row:
                    rows.append(row)
                if err:
                    errors.append(err)
                    print("ERROR", err[0], err[1].splitlines()[-1], flush=True)
                if done_n % max(1, n // 40) == 0 or done_n == n:
                    el = time.time() - t0
                    print(f"[{done_n}/{n}] {el:.0f}s elapsed, {done_n / el:.2f} ep/s, eta {(n - done_n) * el / done_n:.0f}s", flush=True)
    except (BrokenProcessPool, RuntimeError, KeyboardInterrupt) as e:
        print(f"ABORTED after {done_n}/{n} new episodes: {type(e).__name__}: {e}", flush=True)
        for f in pending:
            f.cancel()
        for proc in list(getattr(ex, "_processes", {}).values()):
            proc.kill()
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    ex.shutdown()
    idx = pd.DataFrame(rows).sort_values("seed")
    idx.to_parquet(out / f"index_{a.seed0}.parquet")
    if errors:
        with open(out / f"errors_{a.seed0}.json", "w") as f:
            json.dump(errors, f, indent=1)
    print(idx.groupby(["category", "outcome"]).size().to_string())
    print(f"done: {len(rows)} episodes, {len(errors)} errors, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
