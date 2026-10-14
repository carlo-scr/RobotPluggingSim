# %% [markdown]
# # 06 · Latent world model + conformal failure monitor
#
# This reproduces the lab's detection pipeline (Ward, Ho et al. 2026) on sim data.
#
# 1. **Encoder**: frames → compact latents. Locally a small conv autoencoder trained on nominal
#    frames, with the reconstruction loss weighted towards edges so the thin cable and the plug
#    survive. On GPU nodes, optionally the frozen **Cosmos tokenizer** (`CosmosEncoder`, the same
#    CI16x16 model as `GaugeFailClassification`).
# 2. **Probabilistic latent WM**: an ensemble of history-conditioned Gaussian next-step models over
#    `[latents of the 4 gripper cameras, TCP poses + grip widths]`, trained on **nominal data only**
#    and early-stopped on held-out nominal episodes. Ablations: cameras only, gripper state only.
# 3. **Monitor**: per-step non-conformity scores (one-step NLL, predicted σ, ensemble disagreement,
#    kNN state novelty), their trajectory max, and a **split-conformal threshold** on held-out nominal
#    episodes (false-alarm rate ≤ α on nominal). Scored against the *physical* outcome, with the lead
#    time before the failure is complete.
# 4. **Three-way decision** (Ho et al.): at the alarm, does a failure-WM (fine-tuned on known
#    failures) explain the data better than the success-WM? → *nominal / known failure / OOD*.
#
# Ground-truth labels come from the sim (category, physical outcome, failure mode, drop/latch/phase
# times), so every number below is checkable.

# %%
# @bootstrap

# %%
import os, sys, pathlib, subprocess, time, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from plugsim.learn.data import list_episodes, iter_episodes, split_by_category
from plugsim.learn.encoders import train_ae, save_ae, load_ae, CosmosEncoder
from plugsim.learn.wm import train_wm, wm_features, score_episode, save_wm, load_wm, PROPRIO_IDX
from plugsim.tools import run_live
from plugsim.learn.monitor import (evaluate_monitor, combined_trace, summarize, headline, conformal_threshold, three_way, smooth,
                                   failure_time, window_mean, KNNNovelty)

SCALE = os.environ.get("PLUGSIM_SCALE", "local")
NCPU = max(1, (os.cpu_count() or 2) - 1)
P = {
    "local": dict(n=360, workers=min(6, NCPU), ae_steps=2500, wm_steps=3000, latent=32, hw=(72, 128), device="mps" if torch.backends.mps.is_available() else "cpu"),
    "a100": dict(n=1500, workers=NCPU, ae_steps=15000, wm_steps=20000, latent=48, hw=(72, 128), device="cuda"),  # Colab A100
    "h100": dict(n=6000, workers=NCPU, ae_steps=20000, wm_steps=40000, latent=64, hw=(144, 256), device="cuda"),
}[SCALE]
P.update(ae_edge=10.0, min_log_std=-6.0, fwm_steps=P["wm_steps"] // 4)  # edge-weighted AE loss, WM std floor, failure-WM fine-tuning
CAMS = ["right_gripper_front", "right_gripper_rear", "left_gripper_front", "left_gripper_rear"]
DATA, RUNS = ROOT / "data" / f"wm_{SCALE}", ROOT / "runs"
RUNS.mkdir(exist_ok=True)
USE_COSMOS = os.environ.get("COSMOS_CKPT")  # set to the checkpoint dir on a GPU node to use Cosmos latents
AE_CKPT, FEATS = RUNS / "wm_ae.pt", RUNS / f"wm_feats_{SCALE}{'_cosmos' if USE_COSMOS else ''}.pt"
ALPHA, DT = 0.1, 0.05  # conformal false-alarm level; control step [s]
SEED0 = 100000  # episode seeds of this dataset (the generator writes index_<seed0>.parquet when done)


def cached(target, *deps, cfg=None):
    """Every expensive step below caches to runs/ and is reused when its file is newer than its inputs (like make)
    and was built with the same settings `cfg`, so after a kernel restart "Run All" only redoes what is missing
    or changed. RETRAIN=1 forces a rebuild."""
    side = target.with_name(target.name + ".json")
    return (target.exists() and not os.environ.get("RETRAIN") and all(target.stat().st_mtime > d.stat().st_mtime for d in deps if d.exists())
            and (cfg is None or (side.exists() and json.loads(side.read_text()) == json.loads(json.dumps(cfg)))))


def mark(target, cfg):  # record the settings a cached file was built with
    target.with_name(target.name + ".json").write_text(json.dumps(cfg))


print(SCALE, P, "| cosmos:", bool(USE_COSMOS))

# %% [markdown]
# ## 1 · Dataset: nominal + known failures + OOD, recorded at training resolution

# %%
if not (DATA / f"index_{SEED0}.parquet").exists():
    cfg = json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})
    t = time.time()
    # streams progress; resumes if interrupted (re-run this cell); aborts with a message if workers get stuck
    run_live([sys.executable, "-u", str(ROOT / "scripts/generate_dataset.py"), "--out", str(DATA), "--n", str(P["n"]), "--workers", str(P["workers"]),
              "--mix", "nominal:0.55,known_failure:0.25,ood:0.2", "--cameras", ",".join(CAMS), "--no-states", "--cfg", cfg, "--seed0", str(SEED0)])
    print(f"({time.time() - t:.0f}s)")
idx = list_episodes(DATA)
splits = split_by_category(idx)
print({k: len(v) for k, v in splits.items()})
pd.crosstab(idx.category, idx.outcome)

# %% [markdown]
# ## 2 · Encoder
# The autoencoder is trained on frames from nominal training episodes only (every 3rd frame of each
# camera, pooled across the four cameras). A plain MSE loss blurs the cable and the plug away: they
# cover a few pixels, while the table and the gripper fingers dominate the loss. So the loss is
# weighted towards edges (`edge_weights`: up to 1 + `ae_edge` on high-gradient pixels).

# %%
AE_CFG = dict(latent=P["latent"], steps=P["ae_steps"], edge=P["ae_edge"], hw=list(P["hw"]))
if USE_COSMOS:
    enc = CosmosEncoder(USE_COSMOS, device=P["device"])
else:
    if cached(AE_CKPT, cfg=AE_CFG):
        enc = load_ae(AE_CKPT, device=P["device"])
        print("loaded", AE_CKPT)
    else:
        frames = np.concatenate([np.concatenate([e["frames"][c][::3] for c in CAMS])
                                 for _, e in iter_episodes(splits["nominal_train"].path, CAMS) if e is not None])
        print("AE training frames:", frames.shape)
        enc, ae_hist = train_ae(frames, latent=P["latent"], steps=P["ae_steps"], device=P["device"], edge=P["ae_edge"])
        save_ae(enc, AE_CKPT); mark(AE_CKPT, AE_CFG)
        del frames
    ep = next(e for _, e in iter_episodes(splits["nominal_test"].path, CAMS) if e is not None)
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
def save_feats():  # write-then-rename, so an interrupted save never leaves a corrupt cache
    torch.save(dict(feats=feats, anns=anns), FEATS.with_suffix(".tmp")); FEATS.with_suffix(".tmp").replace(FEATS)


t = time.time()
feats, anns, bad = {}, {}, []
if cached(FEATS, AE_CKPT):  # features of the current encoder from an earlier (possibly interrupted) run
    ck = torch.load(FEATS, weights_only=False)
    feats, anns = ck["feats"], ck["anns"]
todo = [p for p in idx.path if p not in feats]
print(f"{len(feats)} episodes cached, {len(todo)} to encode")
for i, (p, ep) in enumerate(iter_episodes(todo, CAMS)):  # retries, then skips corrupt / incomplete episodes
    if ep is None:
        bad.append(p)
        continue
    z = np.concatenate([enc.encode(ep["frames"][c]) for c in CAMS], 1)
    feats[p] = dict(x=wm_features(z, ep["state"]), action=ep["action"])
    anns[p] = ep["ann"]
    if (i + 1) % 250 == 0:
        print(f"  encoded {i + 1}/{len(todo)} ({time.time() - t:.0f}s)", flush=True)
        save_feats()  # checkpoint: a restart resumes from here
if len(todo) > len(bad):
    save_feats()
# drop unreadable episodes (and cached ones no longer in the dataset) everywhere downstream
idx = idx[idx.path.isin(list(feats))]
splits = {k: v[v.path.isin(list(feats))] for k, v in splits.items()}
feats = {p: feats[p] for p in idx.path}
print(f"{len(feats)} episodes ready in {time.time() - t:.0f}s ({len(bad)} unreadable skipped); x dim = {next(iter(feats.values()))['x'].shape[1]}")

# %% [markdown]
# ## 4 · World models
# Three success-WMs on nominal data that differ only in their input: camera latents + gripper state
# (the full model), camera latents only, gripper state only (TCP poses and widths). The ablation
# shows what the cameras add. Each holds out 10 % of its training episodes and keeps the weights
# with the best held-out NLL.
#
# The failure-WM is the full success-WM **fine-tuned** on 60 % of the known-failure training episodes
# (15 % for validation). Starting from the success-WM keeps what it knows about nominal motion (the
# failure episodes are nominal most of the time) and keeps the same normalisation, so the two
# models' NLLs can be compared step by step. The last 25 % calibrate the three-way decision (§ 7).

# %%
L = next(iter(feats.values()))["x"].shape[1] - len(PROPRIO_IDX)  # latent dims of the 4 cameras
VARIANTS = {"cameras+gripper": slice(None), "cameras": slice(0, L), "gripper": slice(L, None)}
rng = np.random.default_rng(0)
nom_tr = [str(p) for p in rng.permutation(splits["nominal_train"].path.tolist())]
nom_val, nom_fit = nom_tr[: len(nom_tr) // 10], nom_tr[len(nom_tr) // 10:]
kf_tr = [str(p) for p in rng.permutation(splits["known_failure_train"].path.tolist())]
f_fit, f_val, f_cal = kf_tr[: int(0.6 * len(kf_tr))], kf_tr[int(0.6 * len(kf_tr)): int(0.75 * len(kf_tr))], kf_tr[int(0.75 * len(kf_tr)):]
eps = lambda paths, cols=slice(None): [dict(x=feats[p]["x"][:, cols], action=feats[p]["action"]) for p in paths]

EVERY = max(1, P["wm_steps"] // 20)  # held-out checks per run (early stopping granularity)
wms, curves = {}, {}
for name, cols in VARIANTS.items():
    ckpt, cfg = RUNS / f"wm_success_{name}.pt", dict(steps=P["wm_steps"], min_log_std=P["min_log_std"], n_fit=len(nom_fit), every=EVERY)
    if cached(ckpt, FEATS, cfg=cfg):
        wms[name] = load_wm(ckpt, device=P["device"])
        print("loaded", ckpt)
    else:
        print(f"success-WM: {name}")
        wms[name], curves[name] = train_wm(eps(nom_fit, cols), val=eps(nom_val, cols), steps=P["wm_steps"], min_log_std=P["min_log_std"],
                                           device=P["device"], log_every=EVERY)
        save_wm(wms[name], ckpt); mark(ckpt, cfg)  # saved right away, so a crash in the next one keeps it
wm_s = wms["cameras+gripper"]
ckpt, cfg = RUNS / "wm_failure.pt", dict(steps=P["fwm_steps"], n_fit=len(f_fit), every=max(1, P["fwm_steps"] // 20))
if cached(ckpt, RUNS / "wm_success_cameras+gripper.pt", cfg=cfg):
    wm_f = load_wm(ckpt, device=P["device"])
    print("loaded", ckpt)
else:
    print("failure-WM: fine-tuned from the full success-WM")
    wm_f, curves["failure"] = train_wm(eps(f_fit), val=eps(f_val), init=wm_s, steps=P["fwm_steps"], lr=3e-4, device=P["device"],
                                       log_every=max(1, P["fwm_steps"] // 20))
    save_wm(wm_f, ckpt); mark(ckpt, cfg)

fig, ax = plt.subplots(1, 1 + bool(curves), figsize=(6 + 6 * bool(curves), 3), squeeze=False)
for name, wm in {**wms, "failure": wm_f}.items():
    if wm.val_hist:
        s, v = zip(*wm.val_hist); ax[0, 0].plot(s, v, marker=".", label=name)
ax[0, 0].set_title("held-out NLL (best weights kept)")
for name, h in curves.items():  # training curves of the models trained in this run
    ax[0, 1].plot(smooth(h, 50), label=name); ax[0, 1].set_title("train NLL")
for a in ax.flat: a.set_xlabel("step"); a.set_yscale("symlog"); a.legend(fontsize=7)
plt.show()

# %% [markdown]
# ## 5 · Monitor: conformal threshold on nominal calibration episodes, evaluated on held-out data
# Every score is calibrated on the nominal calibration episodes (α = 0.1) and evaluated on the test
# splits. Failures count by their **physical** outcome:
# * `failure_AUROC` / `failure_TPR`: episodes that physically failed vs successful nominal ones;
# * `warned_before_failure`: share of failures flagged *before* they were complete (the drop, the
#   lift without the plug, the release without a latch, the latch in the wrong port);
# * `recovered_alarm_rate`: alarms on injected failures the robot recovered from (e.g. a bump it
#   shrugged off). An anomaly detector *should* see these; a failure predictor should not;
# * `OOD_*`: all OOD episodes (most still succeed: the shift is visual or mild).
#
# Scores: `nll` (one-step NLL of what happened), `alea` (predicted σ, the paper's WM uncertainty),
# `epi` (ensemble disagreement), `nov` (kNN state novelty: catches shifts that move the latents but
# not their deltas, which the NLL of the *next-step change* is blind to), and `nll+nov` (alarm when
# either crosses its α/2 threshold: same false-alarm guarantee).
#
# Baseline for comparison, the first A100 run of this notebook (2026-10-13: plain-MSE encoder, no
# early stopping, full model with `nll`, alarms counted against the *injected* label): 5 % false
# alarms, 74 % of injected failures and 52 % of OOD episodes flagged; its three-way decision never
# answered "known failure".

# %%
dev = P["device"]
eval_splits = {k: v.path.tolist() for k, v in splits.items() if k in ("nominal_cal", "nominal_test", "known_failure_test", "ood_test")}
need = sorted(set(sum(eval_splits.values(), [])) | set(f_cal))
t = time.time()
traces = {name: {p: score_episode(wm, feats[p]["x"][:, VARIANTS[name]], feats[p]["action"], device=dev) for p in need} for name, wm in wms.items()}
knn = KNNNovelty([feats[p]["x"] for p in nom_fit], device=dev)
for p in need:
    traces["cameras+gripper"][p]["nov"] = knn.score(feats[p]["x"])[1:]  # novelty of the state each step leads to
print(f"scored {len(need)} episodes in {time.time() - t:.0f}s")

CONFIGS = [(v, k) for v in VARIANTS for k in ("nll", "alea", "epi")] + [("cameras+gripper", "nov"), ("cameras+gripper", ("nll", "nov"))]
results, norms, rows = {}, {}, []
for v, k in CONFIGS:
    label = "+".join(k) if isinstance(k, tuple) else k
    results[(v, label)], norms[(v, label)] = evaluate_monitor(traces[v], anns, eval_splits, key=k, alpha=ALPHA)
    rows.append(dict(model=v, score=label, **headline(results[(v, label)])))
pd.DataFrame(rows).set_index(["model", "score"]).round(2)

# %% [markdown]
# Per failure type and OOD family, for the full model with `nll+nov`. `alarm_on_failures` is the TPR
# on the episodes of the group that physically failed, `alarm_on_successes` the alarm rate on those
# that did not (for `nominal`: the false-alarm rate).

# %%
MAIN = ("cameras+gripper", "nll+nov")
df, norm = results[MAIN], norms[MAIN]
summarize(df).round(2)

# %%
def symlog_bins(v, n=40):  # histogram bins evenly spaced on a symlog axis (the NLL spans many decades)
    f = lambda x: np.sign(x) * np.log10(1 + np.abs(x))
    y = np.linspace(f(np.min(v)), f(np.max(v)), n)
    return np.sign(y) * (10 ** np.abs(y) - 1)


fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, key in zip(ax, ("nll", "nov", "nll+nov")):
    d = results[("cameras+gripper", key)]
    bins = symlog_bins(d.score)
    for cat, col in (("nominal", "tab:green"), ("known_failure", "tab:red"), ("ood", "tab:purple")):
        a.hist(d[d.category == cat].score, bins=bins, alpha=0.5, color=col, label=cat)
    a.axvline(1, color="k", ls="--", label=f"conformal threshold (α={ALPHA})")
    a.set_xscale("symlog", linthresh=1); a.set_xlabel("episode max score (1 = threshold)"); a.set_title(key); a.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6 · What a detection looks like: score over time
# Normalised `nll+nov` score (1 = conformal threshold) for three nominal episodes (green) and three
# failures of each kind; dotted: the moment the failure is complete.

# %%
tr = traces["cameras+gripper"]
fig, ax = plt.subplots(1, 2, figsize=(15, 3.6), sharey=True)
for a, mode in zip(ax, ("dropped", "not_seated")):
    for p in eval_splits["nominal_test"][:3]:
        c = combined_trace(tr[p], norm); a.plot(np.arange(len(c)) * DT, c, color="tab:green", alpha=0.6)
    for p in [p for p in eval_splits["known_failure_test"] if anns[p]["failure_mode"] == mode][:3]:
        c = combined_trace(tr[p], norm); (l,) = a.plot(np.arange(len(c)) * DT, c)
        a.axvline(failure_time(anns[p]), color=l.get_color(), ls=":", lw=1.2)
    a.axhline(1, color="k", ls="--", label="conformal threshold")
    a.set_yscale("symlog", linthresh=1); a.set_xlabel("t [s]"); a.set_title(f"{mode} (dotted: failure complete)")
ax[0].set_ylabel("normalised score"); ax[0].legend(); plt.show()

# %% [markdown]
# ## 7 · Three-way decision: nominal / known failure / OOD
# An alarmed episode is a **known failure** if its anomaly looks like those in the known-failure
# training episodes, else **OOD**. Both rules below look at the most anomalous moment t* (± 1 s) and
# are calibrated on the alarmed held-out known failures (`f_cal`): the threshold is their conformal
# (1 − α) quantile, so about α of known failures are sent to OOD.
# * **failure-WM ratio** (Ho et al.): r = mean NLL under the failure-WM − under the success-WM (same
#   normalisation, so a log-likelihood ratio); known failure if r ≤ τ.
# * **kNN to known failures**: distance from the states around t* to the states around the anomaly
#   peaks of the alarmed known-failure training episodes; known failure if ≤ τ.
#
# The ratio has a built-in weakness: fine-tuned on failures, the failure-WM becomes more tolerant of
# *any* surprise, so it also "explains" OOD anomalies better than the success-WM. Truth is the
# episode category (an injected failure the robot recovered from is still a known failure type).
# Rows: truth; columns: decision.

# %%
W = int(1.0 / DT)
around = lambda x, i: x[max(0, i - W): i + W + 1]
ct = {p: combined_trace(tr[p], norm) for p in need}
nll_f = {p: score_episode(wm_f, feats[p]["x"], feats[p]["action"], device=dev)["nll"] for p in need}

ref = []  # states around the anomaly peak of every alarmed known-failure training episode
for p in f_fit + f_val:
    c = combined_trace(dict(nll=score_episode(wm_s, feats[p]["x"], feats[p]["action"], device=dev)["nll"], nov=knn.score(feats[p]["x"])[1:]), norm)
    if c.max() > 1:
        ref.append(around(feats[p]["x"], int(np.argmax(c))))
kf_knn = KNNNovelty(ref, stride=1, device=dev)

RULES = {
    "failure-WM ratio": lambda p: window_mean(nll_f[p], int(np.argmax(ct[p])), W) - window_mean(tr[p]["nll"], int(np.argmax(ct[p])), W),
    "kNN to known failures": lambda p: float(kf_knn.score(around(feats[p]["x"], int(np.argmax(ct[p])))).min()),
}
test = [p for s in ("nominal_test", "known_failure_test", "ood_test") for p in eval_splits[s]]
truth = pd.Series([anns[p]["category"] for p in test], name="truth")
tabs = {}
for name, r in RULES.items():
    tau_k = conformal_threshold([r(p) for p in f_cal if ct[p].max() > 1], ALPHA)
    d = pd.DataFrame(dict(truth=truth, pred=[three_way(ct[p].max(), r(p), 1.0, tau_k) for p in test]))
    tabs[name] = pd.crosstab(d.truth, d.pred, normalize="index").reindex(columns=["nominal", "known_failure", "ood"], fill_value=0)
    a = d[(d.pred != "nominal") & (d.truth != "nominal")]
    print(f"{name:22s} τ = {tau_k:.3g}  ({len(ref)} reference failures)   alarmed known failures / OOD sorted correctly: {(a.truth == a.pred).mean():.0%}")
cm = pd.concat(tabs, axis=1).round(2)
cm["n"] = truth.value_counts()
cm
