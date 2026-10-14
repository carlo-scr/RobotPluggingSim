"""Load generated episodes (see plugsim.datagen) into arrays for learning.

Observation pieces used by the learners:
  state  (44)  proprioception, exactly what the real robot logs
  task   (30)  privileged task features (a stand-in for perception, used by the low-dim "state" policy):
               plug pose (7), plug tip (3), target port pose (7), grasped flag (1), and relative
               features that expose mm-level errors: grasp point - TCP per arm (6), plug tip and plug
               axis in the target-port frame (6)
  frames       uint8 camera images (generated at training resolution via the dataset --cfg)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIM = 30


def list_episodes(root) -> pd.DataFrame:
    """Concatenate all index_*.parquet files of a dataset folder."""
    root = Path(root)
    idx = [pd.read_parquet(p) for p in sorted(root.glob("index_*.parquet"))]
    if not idx:
        raise FileNotFoundError(f"no index_*.parquet in {root}")
    return pd.concat(idx, ignore_index=True).sort_values("seed").reset_index(drop=True)


def _quat_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _task(plug_pos, plug_quat, tip, port_pos, port_quat, grasped, tcp_r, tcp_l, grasp_off):
    Rp = _quat_mat(plug_quat)  # [T,3,3]
    grasp = plug_pos + Rp[..., :, 0] * grasp_off
    Rport = _quat_mat(port_quat)
    rel_tip = np.einsum("ji,tj->ti", Rport, tip - port_pos)
    rel_ax = np.einsum("ji,tj->ti", Rport, Rp[..., :, 0])
    T = len(plug_pos)
    return np.concatenate([plug_pos, plug_quat, tip, np.repeat(np.concatenate([port_pos, port_quat])[None], T, 0),
                           grasped[:, None], grasp - tcp_r, grasp - tcp_l, rel_tip, rel_ax], 1).astype(np.float32)


def _grasp_offset(cfg=None):
    from ..config import EnvCfg

    pl = (cfg or EnvCfg()).plug
    return pl.boot_len + pl.body_len - pl.grasp_from_tip


def task_features_from_columns(df: pd.DataFrame, ann: dict, state=None) -> np.ndarray:
    st = np.stack(df["observation.state"].values) if state is None else state
    return _task(np.stack(df["priv.plug_pos"].values), np.stack(df["priv.plug_quat"].values), np.stack(df["priv.tip_pos"].values),
                 np.asarray(ann["target_port_pos"]), np.asarray(ann["target_port_quat"]),
                 (df["priv.grasped_by"].values >= 0).astype(np.float32), st[:, 14:17], st[:, 36:39], _grasp_offset())


def task_features_from_env(env) -> np.ndarray:
    pr = env.privileged()
    pp, pq = env.port_pose(env.scenario.port_idx)
    st = env.observe()[None]
    return _task(pr["plug_pos"][None], pr["plug_quat"][None], pr["tip_pos"][None], pp, pq,
                 np.array([float(pr["grasped_by"] is not None)]), st[:, 14:17], st[:, 36:39], _grasp_offset(env.cfg))[0]


def load_episode_arrays(path, cams=(), frames=True):
    """One episode -> dict(state [T,44], action [T,16], task [T,18], frames {cam: [T,H,W,3]}, ann)."""
    import imageio.v2 as imageio

    path = Path(path)
    df = pd.read_parquet(path / "episode.parquet")
    ann = json.load(open(path / "annotations.json"))
    out = dict(
        state=np.stack(df["observation.state"].values).astype(np.float32),
        action=np.stack(df["action"].values).astype(np.float32),
        ee_target=np.stack(df["action.ee_target"].values).astype(np.float32),
        action_expert=np.stack(df["action.expert"].values).astype(np.float32) if "action.expert" in df else None,
        task=task_features_from_columns(df, ann),
        depth=df["priv.insert_depth"].values.astype(np.float32),
        phase=df["priv.phase"].values,
        ann=ann,
        frames={},
    )
    if frames:
        for c in cams:
            out["frames"][c] = np.stack(imageio.mimread(path / f"{c}.mp4", memtest=False))[: len(df)]
    return out


def try_load_episode(path, cams=(), frames=True, retries=2, wait_s=2.0):
    """load_episode_arrays that retries (e.g. Drive/FUSE hiccups) and returns None for unreadable or
    incomplete episodes (corrupt video, fewer frames than steps) instead of raising."""
    import time as _t

    err = None
    for _ in range(retries + 1):
        try:
            ep = load_episode_arrays(path, cams, frames)
            T = len(ep["state"])
            short = [c for c in cams if frames and len(ep["frames"][c]) < T]
            if short:
                raise ValueError(f"video shorter than episode for {short}")
            return ep
        except Exception as e:  # noqa: BLE001
            err = e
            _t.sleep(wait_s)
    print(f"skipping unreadable episode {Path(path).name}: {type(err).__name__}: {str(err).splitlines()[0][:120]}", flush=True)
    return None


def iter_episodes(paths, cams=(), frames=True, workers=8):
    """try_load_episode over paths, decoding up to 2 * workers episodes ahead in threads (video decoding
    runs in ffmpeg subprocesses, so threads parallelise it). Yields (path, episode or None) in order."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as ex:
        q = deque()
        for p in paths:
            q.append((p, ex.submit(try_load_episode, p, cams, frames)))
            if len(q) > 2 * workers:
                p0, fut = q.popleft()
                yield p0, fut.result()
        while q:
            p0, fut = q.popleft()
            yield p0, fut.result()


def load_many(paths, cams=(), frames=True, progress=True):
    eps = []
    for i, p in enumerate(paths):
        eps.append(load_episode_arrays(p, cams, frames))
        if progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(paths)}", flush=True)
    return eps


class Normalizer:
    """Per-dimension min/max scaling to [-1, 1] (constant dims are left centred). `robust` uses the
    0.5/99.5 percentiles instead, so rare large values don't squash the common range."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    def fit(self, X, robust=False):
        X = np.asarray(X, np.float32).reshape(-1, X.shape[-1])
        if robust:
            self.lo, self.hi = np.percentile(X, 0.5, 0).astype(np.float32), np.percentile(X, 99.5, 0).astype(np.float32)
        else:
            self.lo, self.hi = X.min(0), X.max(0)
        return self

    def scale(self):
        return np.where(self.hi - self.lo > 1e-6, (self.hi - self.lo) / 2, 1.0).astype(np.float32)

    def mid(self):
        return ((self.hi + self.lo) / 2).astype(np.float32)

    def norm(self, X):
        return (X - self.mid()) / self.scale()

    def denorm(self, Xn):
        return Xn * self.scale() + self.mid()

    def state_dict(self):
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist())

    @staticmethod
    def from_state(d):
        return Normalizer(np.asarray(d["lo"], np.float32), np.asarray(d["hi"], np.float32))


def split_by_category(idx: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed=0):
    """Nominal episodes -> train / cal / test; failures and OOD -> 50/50 train(failure-WM)/test."""
    rng = np.random.default_rng(seed)
    out = {}
    nom = idx[idx.category == "nominal"].sample(frac=1, random_state=int(rng.integers(1 << 30)))
    n = len(nom)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    out["nominal_train"], out["nominal_cal"], out["nominal_test"] = nom.iloc[:a], nom.iloc[a:b], nom.iloc[b:]
    for cat in ("known_failure", "ood"):
        d = idx[idx.category == cat].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        h = len(d) // 2
        out[f"{cat}_train"], out[f"{cat}_test"] = d.iloc[:h], d.iloc[h:]
    return out
