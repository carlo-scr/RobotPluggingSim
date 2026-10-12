# %% [markdown]
# # 06 · Latent world model + conformal failure monitor
#
# This reproduces the lab's detection pipeline (Ward, Ho et al. 2026) on sim data.
#
# 1. **Encoder**: frames → compact latents. Locally a small conv autoencoder trained on nominal
#    frames. On GPU nodes, the frozen **Cosmos tokenizer** (`CosmosEncoder`, the same CI16x16
#    model as `GaugeFailClassification`).
# 2. **Probabilistic latent WM**: an ensemble of history-conditioned Gaussian next-step models over
#    `[latents of the 4 gripper cameras, TCP poses + grip widths]`, trained on **nominal data only**.
# 3. **Monitor**: a per-step non-conformity score (predicted σ = "WM uncertainty", or one-step NLL),
#    its trajectory max, and a **split-conformal threshold** on held-out nominal episodes
#    (false-alarm rate ≤ α on nominal), then detection time and lead time before the failure.
# 4. **Three-way decision** (Ho et al.): success-WM vs a failure-WM trained on known failures gives
#    *nominal / known failure / OOD*.
#
# Ground-truth labels come from the sim (category, physical outcome, failure mode, drop/injection
# times), so every number below is checkable.

# %%
# @bootstrap

# %%
import os, sys, pathlib, subprocess, time, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from plugsim.learn.data import list_episodes, load_episode_arrays, split_by_category
from plugsim.learn.encoders import train_ae, save_ae, CosmosEncoder
from plugsim.learn.wm import train_wm, wm_features, score_episode, save_wm
from plugsim.learn.monitor import evaluate_monitor, summarize, auroc, conformal_threshold, three_way, smooth, event_time

SCALE = os.environ.get("PLUGSIM_SCALE", "local")
NCPU = max(1, (os.cpu_count() or 2) - 1)
P = {
    "local": dict(n=360, workers=min(6, NCPU), ae_steps=2500, wm_steps=3000, latent=32, hw=(72, 128), device="mps" if torch.backends.mps.is_available() else "cpu"),
    "a100": dict(n=1500, workers=NCPU, ae_steps=15000, wm_steps=20000, latent=48, hw=(72, 128), device="cuda"),  # Colab A100
    "h100": dict(n=6000, workers=NCPU, ae_steps=20000, wm_steps=40000, latent=64, hw=(144, 256), device="cuda"),
}[SCALE]
CAMS = ["right_gripper_front", "right_gripper_rear", "left_gripper_front", "left_gripper_rear"]
DATA, RUNS = ROOT / "data" / f"wm_{SCALE}", ROOT / "runs"
RUNS.mkdir(exist_ok=True)
USE_COSMOS = os.environ.get("COSMOS_CKPT")  # set to the checkpoint dir on a GPU node to use Cosmos latents
print(SCALE, P, "| cosmos:", bool(USE_COSMOS))

# %% [markdown]
# ## 1 · Dataset: nominal + known failures + OOD, recorded at training resolution

# %%
if not (DATA / "index_0.parquet").exists():
    cfg = json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})
    t = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/generate_dataset.py"), "--out", str(DATA), "--n", str(P["n"]), "--workers", str(P["workers"]),
                        "--mix", "nominal:0.55,known_failure:0.25,ood:0.2", "--cameras", ",".join(CAMS), "--no-states", "--cfg", cfg, "--seed0", "100000"],
                       capture_output=True, text=True)
    print(r.stdout.splitlines()[-1] if r.stdout else r.stderr[-2000:], f"({time.time() - t:.0f}s)")
idx = list_episodes(DATA)
splits = split_by_category(idx)
print({k: len(v) for k, v in splits.items()})
pd.crosstab(idx.category, idx.outcome)

# %% [markdown]
# ## 2 · Encoder
# The autoencoder is trained on frames from nominal training episodes only. Every 3rd frame of each
# camera is pooled across all four cameras.

# %%
if USE_COSMOS:
    enc = CosmosEncoder(USE_COSMOS, device=P["device"])
else:
    sub = splits["nominal_train"].path.tolist()
    frames = np.concatenate([np.concatenate([load_episode_arrays(p, CAMS)["frames"][c][::3] for c in CAMS]) for p in sub])
    print("AE training frames:", frames.shape)
    enc, ae_hist = train_ae(frames, latent=P["latent"], steps=P["ae_steps"], device=P["device"])
    save_ae(enc, RUNS / "wm_ae.pt")
    del frames
    ep = load_episode_arrays(splits["nominal_test"].path.iloc[0], CAMS)
    x = torch.as_tensor(np.stack([ep["frames"][c][60] for c in CAMS])).permute(0, 3, 1, 2).float().to(P["device"]) / 255
    with torch.no_grad():
        rec = enc.decode(enc.forward_latent(x)).permute(0, 2, 3, 1).cpu().numpy()
    fig, ax = plt.subplots(2, 4, figsize=(14, 4.2))
    for i in range(4):
        ax[0, i].imshow(ep["frames"][CAMS[i]][60]); ax[1, i].imshow(rec[i]); ax[0, i].set_title(CAMS[i], fontsize=8)
    for a in ax.flat: a.axis("off")
    fig.suptitle("held-out nominal frame (top) and AE reconstruction (bottom)"); plt.show()

# %% [markdown]
# ## 3 · Encode every episode → WM features

# %%
t = time.time()
feats, anns = {}, {}
for p in idx.path:
    ep = load_episode_arrays(p, CAMS)
    z = np.concatenate([enc.encode(ep["frames"][c]) for c in CAMS], 1)
    feats[p] = dict(x=wm_features(z, ep["state"]), action=ep["action"])
    anns[p] = ep["ann"]
print(f"encoded {len(feats)} episodes in {time.time() - t:.0f}s; x dim = {next(iter(feats.values()))['x'].shape[1]}")

# %% [markdown]
# ## 4 · Train the success-WM (nominal only) and a failure-WM (known failures)

# %%
wm_s, h_s = train_wm([feats[p] for p in splits["nominal_train"].path], steps=P["wm_steps"], device=P["device"])
fail_train = [p for p in splits["known_failure_train"].path if anns[p]["outcome"] == "failure"]
wm_f, h_f = train_wm([feats[p] for p in fail_train], steps=P["wm_steps"], device=P["device"])
save_wm(wm_s, RUNS / "wm_success.pt"); save_wm(wm_f, RUNS / "wm_failure.pt")
plt.figure(figsize=(6, 3)); plt.plot(smooth(h_s, 50), label="success-WM"); plt.plot(smooth(h_f, 50), label="failure-WM")
plt.ylabel("train NLL"); plt.xlabel("step"); plt.legend(); plt.show()

# %% [markdown]
# ## 5 · Monitor: conformal threshold on nominal calibration episodes, evaluate on held-out data

# %%
wm_s_cpu = wm_s.cpu()
traces = {p: score_episode(wm_s_cpu, f["x"], f["action"]) for p, f in feats.items()}
eval_splits = {k: v.path.tolist() for k, v in splits.items() if k in ("nominal_cal", "nominal_test", "known_failure_test", "ood_test")}
ALPHA = 0.1
results = {}
for key in ("alea", "nll", "epi"):
    df, tau = evaluate_monitor(traces, anns, eval_splits, key=key, alpha=ALPHA)
    results[key] = (df, tau)
    print(f"score={key:5s} tau={tau:.3f}  nominal FPR={df[df.category == 'nominal'].flagged.mean():.2f}  "
          f"failure TPR={df[df.category == 'known_failure'].flagged.mean():.2f}  OOD TPR={df[df.category == 'ood'].flagged.mean():.2f}")
df, tau = results["nll"]
summarize(df)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, (key, (d, tau)) in zip(ax, results.items()):
    for cat, col in (("nominal", "tab:green"), ("known_failure", "tab:red"), ("ood", "tab:purple")):
        a.hist(d[d.category == cat].score, bins=25, alpha=0.5, color=col, label=cat, density=True)
    a.axvline(tau, color="k", ls="--", label=f"conformal τ (α={ALPHA})")
    a.set_title(f"episode max score: {key}"); a.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6 · What a detection looks like: score over time for a dropped cable

# %%
drop = [p for p in eval_splits["known_failure_test"] if anns[p]["failure_mode"] == "dropped"]
nomi = eval_splits["nominal_test"][:3]
fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
d, tau = results["nll"]
for p in nomi:
    ax.plot(np.arange(len(traces[p]["nll"])) * 0.05, smooth(traces[p]["nll"]), color="tab:green", alpha=0.6)
for p in drop[:3]:
    tr = smooth(traces[p]["nll"]); tt = np.arange(len(tr)) * 0.05
    ax.plot(tt, tr, color="tab:red")
    te = event_time(anns[p])
    if te: ax.axvline(te, color="tab:red", ls=":", lw=1)
ax.axhline(tau, color="k", ls="--", label="conformal threshold")
ax.set_xlabel("t [s]"); ax.set_ylabel("one-step NLL (smoothed)"); ax.set_title("green: nominal · red: dropped cable (dotted: drop time)"); ax.legend(); plt.show()

# %% [markdown]
# ## 7 · Three-way decision: nominal / known failure / OOD

# %%
wm_f_cpu = wm_f.cpu()
key = "nll"
S = {p: smooth(traces[p][key]).max() for p in feats}
F = {p: smooth(score_episode(wm_f_cpu, feats[p]["x"], feats[p]["action"])[key]).mean() for p in feats}
tau_s = conformal_threshold([S[p] for p in eval_splits["nominal_cal"]], ALPHA)
tau_f = np.quantile([F[p] for p in fail_train], 0.9)
rows = []
for split in ("nominal_test", "known_failure_test", "ood_test"):
    for p in eval_splits[split]:
        truth = anns[p]["category"] if anns[p]["category"] != "known_failure" or anns[p]["outcome"] == "failure" else "nominal"
        rows.append(dict(truth=truth, pred=three_way(S[p], F[p], tau_s, tau_f)))
cm = pd.crosstab(pd.DataFrame(rows).truth, pd.DataFrame(rows).pred, normalize="index").round(2)
cm
