# %% [markdown]
# # 03 · Known failures and OOD shifts
#
# The lab's monitors separate **success / known failure / OOD** (Ho et al.), so the generator
# produces all three, with labels that keep *what was injected* separate from *what physically
# happened*:
#
# | field (annotations.json) | meaning |
# |---|---|
# | `category` | how the episode was generated: `nominal`, `known_failure`, `ood` |
# | `injected_failure` | type, severity ∈ [0,1] and the concrete drawn parameters |
# | `outcome` | `success` iff the target port is latched and the gripper has let go |
# | `failure_mode` | physical outcome: `dropped`, `jammed`, `not_seated`, `wrong_port`, `grasp_missed`, … |
# | `injection_time_s`, `drop_times_s`, `latch_time_s` | event times for detection-delay metrics |
#
# **Known failures** (the expert corrupts its own behaviour):
# `drop` (the real dataset's main failure: gripper opens during transport), `weak_grip`,
# `misalign` (biased port estimate, no search), `partial_insert`, `wrong_port`, `misgrasp`
# (onto the plug body or the cable), `bump` (a knock on the gripper).
#
# **OOD shifts** (outside the nominal randomisation ranges): `lighting`, `cable_color`,
# `camera_shift`, `distractors`, `heavy_cable`, `slippery`, `panel_color`. The scripted expert
# uses privileged state, so it mostly still succeeds under OOD. These shifts are meant to break the
# *learned* models (policy, world model) that only see pixels and proprioception.

# %%
# @bootstrap

# %%
import os, sys, pathlib, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from plugsim.env import PlugEnv
from plugsim.datagen import EpisodeSpec, run_episode
from plugsim.failures import FAILURE_TYPES
from plugsim.scenario import OOD_TYPES
from plugsim.tools import image_grid, video_html

SEEDS_PER_CELL = 4         # laptop; 50+ on a GPU node for smooth severity curves
SEVERITIES = (0.2, 0.5, 0.9)

# %% [markdown]
# ## Outcome matrix: success rate per failure type and severity

# %%
env = PlugEnv()
rows, t = [], time.time()
for kind in FAILURE_TYPES:
    for sev in SEVERITIES:
        for s in range(SEEDS_PER_CELL):
            a = run_episode(env, EpisodeSpec(seed=2000 + s, category="known_failure", failure=kind, severity=sev), cameras=[], save_states=False)["annotations"]
            rows.append(dict(kind=kind, severity=sev, success=a["outcome"] == "success", mode=a["failure_mode"]))
for o in OOD_TYPES:
    for s in range(SEEDS_PER_CELL):
        a = run_episode(env, EpisodeSpec(seed=2000 + s, category="ood", ood=[o]), cameras=[], save_states=False)["annotations"]
        rows.append(dict(kind=f"ood:{o}", severity=np.nan, success=a["outcome"] == "success", mode=a["failure_mode"]))
R = pd.DataFrame(rows)
print(f"{len(R)} episodes in {time.time() - t:.0f}s")
M = R[R.kind.isin(FAILURE_TYPES)].pivot_table(index="kind", columns="severity", values="success", aggfunc="mean").loc[list(FAILURE_TYPES)]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(M.values, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(M.columns)), [f"sev {c}" for c in M.columns]); ax.set_yticks(range(len(M.index)), M.index)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center")
ax.set_title("success rate under injected failures"); plt.colorbar(im); plt.show()
print(R[R.kind.str.startswith("ood")].groupby("kind").success.mean().to_string())

# %%
pd.crosstab(R.kind, R["mode"])

# %% [markdown]
# ## What the failures look like
# The active gripper's front camera at the moment the failure shows: the first frame after
# injection, or the last frame for slow failures.

# %%
imgs, titles = [], []
for kind in FAILURE_TYPES:
    ep = run_episode(env, EpisodeSpec(seed=2001, category="known_failure", failure=kind, severity=0.9),
                     cameras=["right_gripper_front", "left_gripper_front"], save_states=False)
    A = ep["annotations"]; cam = f"{A['active_arm']}_gripper_front"
    t = np.asarray(ep["data"]["t"])
    te = (A["drop_times_s"] or [None])[0] or A["injection_time_s"]
    k = len(t) - 1 if te is None or kind in ("partial_insert", "misalign", "wrong_port") else min(len(t) - 1, int(np.searchsorted(t, te)) + 8)
    imgs.append(ep["frames"][cam][k]); titles.append(f"{kind} → {A['failure_mode']}")
image_grid(imgs, titles, cols=4); plt.show()

# %% [markdown]
# ## The canonical failure: a drop during transport

# %%
ep = run_episode(env, EpisodeSpec(seed=2003, category="known_failure", failure="drop", severity=0.6),
                 cameras=["head_front", "right_gripper_front", "left_gripper_front"], save_states=False)
A = ep["annotations"]; cam = f"{A['active_arm']}_gripper_front"
print({k: A[k] for k in ("failure_mode", "injection_time_s", "drop_times_s", "duration_s")})
video_html([np.concatenate([a, b], 1) for a, b in zip(ep["frames"]["head_front"], ep["frames"][cam])], fps=20, width=760)
