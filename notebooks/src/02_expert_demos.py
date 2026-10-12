# %% [markdown]
# # 02 · Scripted expert ("teleoperator") demos
#
# The real dataset was collected by a technician teleoperating WR1. Here a **privileged state
# machine** plays that role (`plugsim/expert.py`):
#
# `approach → descend → close → lift → transport → align → insert → hold → release → retreat → home`
#
# * The arm on the plug's side picks it up with a top-down grasp on the boot.
# * During transport it re-orients the plug to point into the panel. It then **re-measures the
#   plug-in-hand pose** (closed loop) and aligns 20 mm in front of the lit port.
# * Insertion uses the compliant controller mode and a bounded push force. A small visual servo runs
#   until the tip reaches the chamfer.
# * On a jam it **backs off and retries with a spiral search offset** (≤ 3 retries), the way an
#   operator wiggles a stuck connector. These recoveries show up in nominal data.
#
# Nominal variability: per-episode speed (×0.85–1.2), approach height, grasp jitter (σ 1.5 mm),
# perception noise on the port pose (σ 0.3 mm, 0.3°), plus the scenario randomisation from 00.

# %%
# @bootstrap

# %%
import os, sys, pathlib, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from plugsim.env import PlugEnv
from plugsim.datagen import EpisodeSpec, run_episode
from plugsim.tools import video_html, image_grid

N_EPISODES = 20  # laptop; use thousands via scripts/generate_dataset.py on a GPU node

# %% [markdown]
# ## Nominal success rate (physics only, no rendering)

# %%
env = PlugEnv()
rows, t = [], time.time()
for s in range(N_EPISODES):
    a = run_episode(env, EpisodeSpec(seed=1000 + s), cameras=[], save_states=False)["annotations"]
    rows.append(dict(seed=a["spec"]["seed"], outcome=a["outcome"], mode=a["failure_mode"], arm=a["active_arm"],
                     port=a["target_port"], retries=a["retries"], duration_s=a["duration_s"], latch_s=a["latch_time_s"]))
df = pd.DataFrame(rows)
print(f"{len(df)} episodes in {time.time() - t:.0f}s  ->  success {np.mean(df.outcome == 'success'):.0%}, "
      f"mean duration {df.duration_s.mean():.1f}s, episodes with retries: {(df.retries > 0).sum()}")
df.head(10)

# %% [markdown]
# ## One episode in detail
# Recorded with all cameras, so we can look at it. `run_episode` returns per-step state, action,
# and privileged ground truth, which is exactly what gets written to disk in notebook 04.

# %%
cams = ["right_gripper_front", "right_gripper_rear", "left_gripper_front", "left_gripper_rear", "head_front"]
ep = run_episode(env, EpisodeSpec(seed=1003), cameras=cams)
A, D = ep["annotations"], ep["data"]
print({k: A[k] for k in ("outcome", "active_arm", "target_port", "latch_time_s", "grasp_time_s", "retries", "duration_s")})
t = D["t"]
fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
ax[0].plot(t, D["depth"] * 1e3); ax[0].set_ylabel("tip depth [mm]"); ax[0].set_ylim(-80, 25)
ax[1].plot(t, D["lateral"] * 1e3); ax[1].set_ylabel("lateral err [mm]"); ax[1].set_yscale("log")
arm_i = ["right", "left"].index(A["active_arm"])
act = D["action"].reshape(len(t), 2, -1)[:, arm_i]
ax[2].plot(t, act[:, 6], label="grip cmd"); ax[2].plot(t, act[:, 7], label="compliant"); ax[2].plot(t, (D["grasped"] >= 0).astype(float) * 0.9, label="grasped (priv)")
ax[2].legend(loc="right"); ax[2].set_ylabel("cmd")
ax[3].plot(t, D["latched"] >= 0); ax[3].set_ylabel("latched"); ax[3].set_xlabel("t [s]")
for name, te in A["phase_events"]:
    for a_ in ax:
        a_.axvline(te, color="0.85", lw=0.8, zorder=0)
    ax[0].text(te, 20, name, fontsize=7, rotation=90, va="top")
plt.tight_layout(); plt.show()

# %%
side = A["active_arm"]
frames = [np.concatenate([np.concatenate([ep["frames"][f"{side}_gripper_front"][i], ep["frames"][f"{side}_gripper_rear"][i]], 1),
                          np.concatenate([ep["frames"]["head_front"][i], np.zeros_like(ep["frames"]["head_front"][i])], 1)], 0)
          for i in range(len(t))]
video_html(frames, fps=20, width=760)

# %% [markdown]
# ## Episode statistics
# Recorded TCP trajectories of the active arm in a few episodes (top view), and the action
# magnitudes the policy will have to learn.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in range(6):
    e = run_episode(env, EpisodeSpec(seed=1100 + s), cameras=[], save_states=False)
    st = e["data"]["state"]; i = ["right", "left"].index(e["annotations"]["active_arm"])
    tcp = st[:, 22 * i + 14 : 22 * i + 17]
    ax[0].plot(tcp[:, 0], tcp[:, 1], lw=1)
    a = e["data"]["action"].reshape(len(st), 2, -1)[:, i]
    ax[1].hist(np.linalg.norm(a[:, :3], axis=1) * 1e3, bins=40, alpha=.4)
ax[0].axvline(env.cfg.rack.face_x, color="k"); ax[0].set_aspect("equal"); ax[0].set_title("active-arm TCP (top view)"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].set_xlabel("|Δp| per step [mm]"); ax[1].set_title("translation action magnitude"); plt.show()
