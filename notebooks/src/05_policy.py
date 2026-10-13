# %% [markdown]
# # 05 · Base policy: Diffusion Policy trained on the scripted demos
#
# The steering work keeps a strong generative policy *frozen* and steers it at inference time, so
# we first need a policy. This is a **Diffusion Policy** (Chi et al. 2023): a 1D-conv U-Net over
# action chunks with FiLM conditioning, DDPM training and DDIM sampling (`plugsim/learn/policy.py`).
#
# Observation modes:
# * `state`: proprioception (44) + privileged task features (plug pose, target-port pose, and
#   relative errors). This stands in for perception, trains in minutes on a laptop, and is the
#   **local default**.
# * `image`: proprioception + gripper-camera frames (small CNN with spatial-softmax keypoints). This
#   is the realistic setting, meant for GPU nodes (more demos, long training).
#
# What it took to make BC work on this precision task (each was a 0 % → working change):
# 1. **Predict the clean action chunk** (x0 / "sample" prediction) instead of the noise. With a
#    laptop-scale budget, ε-prediction stayed at cm-level error.
# 2. **Relative targets**: each chunk is a sequence of controller targets relative to the current
#    target, with robust (percentile) scaling, so fine near-goal motions are small, precisely learnt
#    numbers. Absolute world targets or env deltas both failed.
# 3. **DART noise** (Laskey et al. 2017) in the demos: execute expert action + noise and record the
#    expert's clean corrective action, so the data shows how to recover from small errors.
#
# Expect a policy that is clearly *worse* than the privileged expert. That's useful, because it
# leaves room for steering (notebook 07).

# %%
# @bootstrap

# %%
import os, sys, pathlib, subprocess, time, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from plugsim.env import PlugEnv
from plugsim.scenario import sample_scenario
from plugsim.expert import ScriptedExpert
from plugsim.learn.data import list_episodes, load_many
from plugsim.learn.policy import PolicyCfg, train_policy, save_policy, load_policy, PolicyRunner, rollout
from plugsim.tools import video_html

SCALE = os.environ.get("PLUGSIM_SCALE", "local")
NCPU = max(1, (os.cpu_count() or 2) - 1)
OBS = os.environ.get("PLUGSIM_OBS", "state")  # "state" | "image"
P = {
    "local": dict(n_demos=1000, workers=min(7, NCPU), dart=0.0025, steps=10000, bs=256, n_eval=20, obs_mode=OBS,
                  device="mps" if torch.backends.mps.is_available() else "cpu"),
    "a100": dict(n_demos=4000, workers=NCPU, dart=0.0025, steps=60000, bs=1024, n_eval=50, obs_mode=OBS, device="cuda"),  # Colab A100
    "h100": dict(n_demos=10000, workers=NCPU, dart=0.0025, steps=200000, bs=1024, n_eval=200, obs_mode=OBS, device="cuda"),
}[SCALE]
IMG_CAMS = ("right_gripper_front", "left_gripper_front", "head_front")
DATA, RUNS = ROOT / "data" / f"demos_{P['obs_mode']}_{SCALE}", ROOT / "runs"
CKPT = RUNS / f"policy_{P['obs_mode']}.pt"
RUNS.mkdir(exist_ok=True)
print(SCALE, P)

# %% [markdown]
# ## 1 · Demonstrations (nominal scripted expert + DART noise)
# Re-running after a disconnect is cheap: cached demos and a saved checkpoint are reused. If the
# kernel restarted, run **all** cells from the top again (the first cells only re-import).

# %%
print("cached demos:", (DATA / "index_0.parquet").exists(), "| cached policy:", CKPT.exists(), "->", CKPT)
if not (DATA / "index_0.parquet").exists() and not (CKPT.exists() and not os.environ.get("RETRAIN")):
    cams = "none" if P["obs_mode"] == "state" else ",".join(IMG_CAMS)
    cfg_over = json.dumps({"camera.width": 128, "camera.height": 72})
    t = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/generate_dataset.py"), "--out", str(DATA), "--n", str(P["n_demos"]), "--workers", str(P["workers"]),
                        "--mix", "nominal:1", "--cameras", cams, "--dart", str(P["dart"]), "--no-states", "--cfg", cfg_over], capture_output=True, text=True)
    print(r.stdout.splitlines()[-1] if r.stdout else r.stderr[-2000:], f"({time.time() - t:.0f}s)")
NEED_TRAIN = not CKPT.exists() or bool(os.environ.get("RETRAIN"))
if NEED_TRAIN:  # demos are only needed for training
    idx = list_episodes(DATA)
    idx = idx[idx.outcome == "success"]
    eps = load_many(idx.path.tolist(), cams=IMG_CAMS if P["obs_mode"] == "image" else (), frames=P["obs_mode"] == "image", progress=False)
    print(f"{len(eps)} successful demos, {sum(len(e['action']) for e in eps)} steps")

# %% [markdown]
# ## 2 · Train (or load a cached checkpoint)

# %%
cfg = PolicyCfg(obs_mode=P["obs_mode"], cams=IMG_CAMS, action_mode="rel", prediction="sample", ddim_eta=0.0)
if not NEED_TRAIN:
    pol = load_policy(CKPT)
    print("loaded", CKPT)
else:
    t = time.time()
    pol, hist = train_policy(eps, cfg, steps=P["steps"], bs=P["bs"], lr=1e-3, device=P["device"], log_every=max(1, P["steps"] // 10))
    save_policy(pol, CKPT)
    print(f"trained in {time.time() - t:.0f}s -> {CKPT}")
    plt.figure(figsize=(6, 3)); plt.semilogy(pd.Series(hist).rolling(100).mean()); plt.xlabel("step"); plt.ylabel("x0 MSE (normalised)"); plt.show()
pol = pol.cpu().eval()
print(f"{sum(p.numel() for p in pol.parameters()) / 1e6:.2f} M parameters")

# %% [markdown]
# ## 3 · Closed-loop evaluation on held-out scenarios (vs. the privileged expert)

# %%
env = PlugEnv(render=False)
runner = PolicyRunner(pol, env, seed=0)
rows, t0 = [], time.time()
for s in range(P["n_eval"]):
    sc = sample_scenario(env.cfg, 900_000 + s)
    r = rollout(env, runner, sc)
    pr = env.privileged()
    env.reset(sc); ex = ScriptedExpert(env, seed=900_000 + s)
    while not ex.done and env.time < 16: env.step(ex.act())
    rows.append(dict(seed=900_000 + s, policy=r["success"], expert=env.latched_port == sc.port_idx, policy_latch_s=r["latch_time"], port=sc.port_idx))
R = pd.DataFrame(rows)
print(f"policy success {R.policy.mean():.0%}  |  expert success {R.expert.mean():.0%}  ({time.time() - t0:.0f}s)")
R.head(10)

# %% [markdown]
# ## 4 · One policy rollout

# %%
envr = PlugEnv()
runner_r = PolicyRunner(pol, envr, seed=1)
seed = int(R[R.policy].seed.iloc[0]) if R.policy.any() else 900_000
sc = sample_scenario(envr.cfg, seed)
envr.reset(sc)
frames = []
t_latch = None
while envr.time < 16:
    envr.step(runner_r.act())
    if envr.step_count % 2 == 0:
        arm = "right" if envr.privileged()["grasp_pos"][1] < 0 else "left"
        frames.append(np.concatenate([envr.render_free(lookat=(0.4, 0, 0.95), distance=1.5, azimuth=90, elevation=-30, width=384, height=216),
                                      envr.render(f"{arm}_gripper_front", 384, 216)], 1))
    if envr.latched_port >= 0 and t_latch is None: t_latch = envr.time
    if t_latch and envr.time - t_latch > 1: break
print("latched:", envr.latched_port, "target:", sc.port_idx)
video_html(frames, fps=10, width=760)
