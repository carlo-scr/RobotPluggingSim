# %% [markdown]
# # 04 · Dataset generation (laptop → H100 node), format, counterfactual branching
#
# `scripts/generate_dataset.py` runs one MuJoCo env per worker process. Episodes are independent and
# seeded, so throughput scales with CPU cores, and a big run can be sharded across nodes with
# `--seed0`. On a headless GPU node, rendering goes through EGL (`MUJOCO_GL=egl`).
#
# Each episode folder uses the **same layout as the real WR1 data** read by
# `GaugeFailClassification/src/data/complex_datasets/trajectory.py`: `episode.parquet` with an int64-ns
# `timestamp`, `observation.state` and `action`, one `<camera>.mp4` per feed, plus `annotations.json`
# and `feature_report.json`. On top of that there are `priv.*` ground-truth columns and
# `sim_states.npz` (full simulator snapshot per step).

# %%
# @bootstrap

# %%
import os, sys, pathlib, subprocess, json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

OUT = ROOT / "data" / "smoke"   # small local dataset (gitignored)
N, WORKERS = 12, 4

# %% [markdown]
# ## Generate a small dataset locally

# %%
t = time.time()
cmd = [sys.executable, str(ROOT / "scripts" / "generate_dataset.py"), "--out", str(OUT), "--n", str(N),
       "--workers", str(WORKERS), "--mix", "nominal:0.5,known_failure:0.35,ood:0.15"]
r = subprocess.run(cmd, capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if "WARNING" not in l)[-1500:]); print(r.stderr[-500:] if r.returncode else "")
wall = time.time() - t
idx = pd.read_parquet(OUT / "index_0.parquet")
size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
print(f"{N} episodes, {wall:.0f}s wall with {WORKERS} workers, {size_mb / N:.1f} MB/episode")
idx[["episode", "category", "outcome", "failure_mode", "injected", "severity", "ood", "retries", "duration_s"]]

# %% [markdown]
# ## What's in an episode

# %%
from plugsim.datagen import load_episode
ep = load_episode(idx.path.iloc[0], load_frames=True)
df = ep["df"]
print("columns:", list(df.columns))
print("state dims:", len(ep["features"]["observation_state_names"]), "| action dims:", len(ep["features"]["action_names"]))
print("cameras:", ep["features"]["cameras"], "| frames:", {c: v.shape for c, v in ep["frames"].items()})
print(json.dumps({k: ep["annotations"][k] for k in ("category", "outcome", "failure_mode", "injected_failure", "target_port", "latch_time_s", "drop_times_s")}, indent=1, default=str))
df.head(3)

# %% [markdown]
# ### Compatibility with the lab's loaders
# `Trajectory` reads `timestamp` as ns, requires equal spacing, and loads `<label>.mp4` for each label
# in `config.FEED_IMAGE_LABELS_TO_USE`. To train the Cosmos-latent WM on sim data, point it at an
# episode folder and set
# `FEED_IMAGE_LABELS_TO_USE = ["right_gripper_front", ...]`, `STATE_OBS_DIM = 44`, `ACTION_DIM = 16`.

# %%
ts = df["timestamp"].values.astype(np.float64) * 1e-9
assert np.allclose(np.diff(ts), np.diff(ts)[0], atol=1e-6), "timestamps must be equally spaced"
print(f"dt = {np.diff(ts)[0]:.3f}s equally spaced ✓; files:", sorted(p.name for p in pathlib.Path(idx.path.iloc[0]).iterdir()))

# %% [markdown]
# ## Counterfactual branching from a saved state (the RQ1 "oracle" mechanic)
# Rebuild an episode up to a decision point (the start of `insert`), then execute **K candidate
# action chunks** from the identical state. The chunks are the recorded chunk with a lateral offset
# of growing size in a random direction. Then the base policy (the expert, retries off) takes over,
# and we record whether each candidate ends latched. This is the ground truth that imagined-rollout
# scores (WM σ, success-vs-failure WM, …) get ranked against.

# %%
from plugsim.env import PlugEnv
from plugsim.datagen import EpisodeSpec
from plugsim.tools import expert_at, chunk_then_expert, clone_expert, restore_expert
nom = idx[(idx.category == "nominal") & (idx.outcome == "success")].iloc[0]
E = load_episode(nom.path)
spec = EpisodeSpec(**E["annotations"]["spec"])
k = int(np.argmax(E["df"]["priv.phase"].values == "insert"))
env = PlugEnv(render=False)
ex = expert_at(env, spec, k)
assert np.abs(env.observe() - E["df"]["observation.state"][k]).max() < 1e-5  # identical to the recording
snap, exs = env.get_state(), clone_expert(ex)
H, K, rng = 30, 16, np.random.default_rng(0)
arm = ["right", "left"].index(E["annotations"]["active_arm"])
results = []
for i in range(K):
    env.set_state(snap); restore_expert(ex, exs); ex.max_retries = 0
    chunk = E["actions64"][k : k + H].copy()
    r = 4e-3 * i / (K - 1); ang = rng.uniform(0, 2 * np.pi)
    chunk[0, 8 * arm + 1 : 8 * arm + 3] += r * np.array([np.cos(ang), np.sin(ang)])  # lateral jump, kept for the chunk
    ok = chunk_then_expert(env, ex, chunk)
    results.append(dict(candidate=i, offset_mm=round(r * 1e3, 2), direction_deg=round(np.degrees(ang)), success=ok,
                        max_depth_mm=round(env.privileged()["depth"] * 1e3, 1)))
R = pd.DataFrame(results); print(f"oracle success rate over candidates: {R.success.mean():.0%}"); R

# %% [markdown]
# ## Scaling up on an H100 node
#
# ```bash
# # one node (e.g. 2x48-core host with 8xH100): physics on CPU cores, rendering via EGL
# cd RobotPluggingSim && pip install -e .
# MUJOCO_GL=egl python scripts/generate_dataset.py --out /scratch/plugsim/v0 --n 20000 --workers 90
#
# # shard over several nodes / SLURM array: disjoint seed ranges, then concat the index files
# MUJOCO_GL=egl python scripts/generate_dataset.py --out /scratch/plugsim/v0 --n 20000 --seed0 $((SLURM_ARRAY_TASK_ID*20000)) --workers 90
#
# # variants without code changes (EnvCfg overrides), e.g. tighter jack + smaller frames + all cameras
# python scripts/generate_dataset.py --out /scratch/plugsim/tight --n 5000 \
#     --cfg '{"port.clearance": 0.0005, "camera.width": 256, "camera.height": 144}' \
#     --cameras right_gripper_front,right_gripper_rear,left_gripper_front,left_gripper_rear,head_front,chest_left,chest_right
# ```
#
# Physics-only runs (`--cameras none`) are about 4× faster and still give state, actions, labels and
# snapshots. Frames can be rendered later from `sim_states.npz`. The GPUs are mainly for what comes
# next: training policies and the Cosmos-latent world model on these episodes, and batched
# counterfactual scoring.

# %%
per_ep_worker_s = wall * WORKERS / N
for cores in (8, 48, 96):
    print(f"{cores:3d} workers ≈ {cores / per_ep_worker_s:5.1f} episodes/s ≈ {3600 * cores / per_ep_worker_s / 1000:5.1f}k episodes/hour "
          f"≈ {3600 * cores / per_ep_worker_s * size_mb / N / 1000:5.1f} GB/hour  (extrapolated from this machine)")
