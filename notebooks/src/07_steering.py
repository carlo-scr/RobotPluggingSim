# %% [markdown]
# # 07 · Steering a frozen policy with the latent world model
#
# This is the "detect → **steer**" step of the Latent Steering project, run end to end in the sim with
# the same building blocks as the lab's `dnh` code:
#
# 1. **Collect** base-policy rollouts (nominal + physically perturbed). Train a **success-WM** on
#    successful nominal rollouts and a **failure-WM** on failed ones. Both run on the latent features
#    of notebook 06's encoder.
# 2. **RQ1 oracle ranking** (week-1 go/no-go). From saved mid-episode states, sample K chunks, score
#    each with imagined WM rollouts, then **execute every candidate from the restored sim state** and
#    let the base policy finish. Does the imagined score rank candidates by eventual success?
#    (AUROC, steering gain.) The sim's bit-exact save/restore is what makes this cheap.
# 3. **Closed-loop steering**: base policy vs random-of-K vs best-of-K (WM σ) vs best-of-K
#    (success-vs-failure contrast), on nominal and perturbed scenarios.
# 4. **Abstention**: a conformal threshold η on the chosen candidate's score; abstain when no
#    candidate is certified.
# 5. **Learn-then-Test** on the paired harm rate P(base succeeds ∧ steered fails) over steering
#    strength λ (the IROS draft's do-no-harm guarantee).
#
# Requires `runs/policy_state.pt` (notebook 05) and `runs/wm_ae.pt` (notebook 06).

# %%
# @bootstrap

# %%
import os, sys, pathlib, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from plugsim.env import PlugEnv
from plugsim.scenario import sample_scenario
from plugsim.learn.policy import load_policy, PolicyRunner
from plugsim.learn.encoders import load_ae
from plugsim.learn.wm import train_wm, score_episode
from plugsim.learn.monitor import auroc, conformal_threshold
from plugsim.learn.steer import (FeatureTracker, RandomScorer, WMScorer, ContrastScorer, steered_episode,
                                 capture_branches, rq1_ranking, rq1_metrics, ltt_calibrate)

SCALE = os.environ.get("PLUGSIM_SCALE", "local")
P = {"local": dict(n_nom=40, n_pert=24, K=8, n_branch=16, n_eval=16, wm_steps=2500, device="cpu"),
     "a100": dict(n_nom=150, n_pert=100, K=12, n_branch=64, n_eval=40, wm_steps=15000, device="cuda"),  # Colab A100 (~1 h; rollouts are sequential)
     "h100": dict(n_nom=1500, n_pert=1000, K=16, n_branch=300, n_eval=300, wm_steps=40000, device="cuda")}[SCALE]
DEV = P["device"]
PERTURB = ["heavy_cable", "arm_gain_drift", "slippery", "rack_offset"]  # physical shifts that hurt the policy
CAMS = ["right_gripper_front", "right_gripper_rear", "left_gripper_front", "left_gripper_rear"]
RUNS = ROOT / "runs"
pol = load_policy(RUNS / f"policy_{os.environ.get('PLUGSIM_OBS', 'state')}.pt", device=DEV)
ae = load_ae(RUNS / "wm_ae.pt", device=DEV)
env = PlugEnv()
runner = PolicyRunner(pol, env, device=DEV, seed=0)
tracker = FeatureTracker(ae, CAMS, H=2, hw=ae.hw)
print(SCALE, P)

# %% [markdown]
# ## 1 · Base-policy rollouts → success-WM and failure-WM

# %%
def collect(sc, max_s=16.0):
    env.reset(sc); runner.reset(); tracker.reset()
    X, A, t_latch = [], [], None
    while env.time < max_s and not env.unstable:
        X.append(tracker.observe(env))
        a = runner.act(); A.append(a); tracker.record_action(a); env.step(a)
        if env.latched_port >= 0 and t_latch is None: t_latch = env.time
        if t_latch is not None and env.time - t_latch > 1.0: break
    return dict(x=np.asarray(X, np.float32), action=np.asarray(A, np.float32), success=bool(env.latched_port == sc.port_idx))

t0 = time.time()
roll = []
for i in range(P["n_nom"]):
    roll.append(dict(kind="nominal", **collect(sample_scenario(env.cfg, 700_000 + i))))
for i in range(P["n_pert"]):
    o = PERTURB[i % len(PERTURB)]
    roll.append(dict(kind=o, **collect(sample_scenario(env.cfg, 710_000 + i, ood=[o]))))
R = pd.DataFrame([dict(kind=r["kind"], success=r["success"]) for r in roll])
print(f"{len(roll)} rollouts in {time.time() - t0:.0f}s"); print(R.groupby("kind").success.agg(["mean", "count"]).to_string())

# %%
succ_eps = [r for r in roll if r["kind"] == "nominal" and r["success"]]
fail_eps = [r for r in roll if not r["success"]]
wm_s, _ = train_wm(succ_eps, steps=P["wm_steps"], device=DEV, log_every=P["wm_steps"] // 2)
wm_f, _ = train_wm(fail_eps if len(fail_eps) >= 3 else succ_eps, steps=P["wm_steps"], device=DEV, log_every=P["wm_steps"] // 2)
s_alea, s_epi = WMScorer(wm_s, "alea", "wm_sigma", device=DEV), WMScorer(wm_s, "epi", "wm_epistemic", device=DEV)
f_alea = WMScorer(wm_f, "alea", device=DEV)
scorers = [RandomScorer(0), s_alea, s_epi, ContrastScorer(s_alea, f_alea)]
# sanity: one-step NLL under the success-WM separates successful from failed rollouts
mx = [score_episode(wm_s, r["x"], r["action"], device=DEV)["nll"].max() for r in roll]
print(f"success-WM episode-max NLL AUROC (failed vs successful rollouts): {auroc(np.array(mx)[R.success], np.array(mx)[~R.success]):.2f}")

# %% [markdown]
# ## 2 · RQ1 oracle ranking: do imagined scores rank candidate chunks by eventual success?

# %%
t0 = time.time()
br_sc = [sample_scenario(env.cfg, 720_000 + i, ood=[] if i % 2 == 0 else [PERTURB[(i // 2) % len(PERTURB)]]) for i in range(P["n_branch"])]
branches = capture_branches(env, runner, tracker, br_sc, at_replans=(3, 5, 7, 9))
succ, scores = rq1_ranking(env, runner, tracker, scorers, branches, K=P["K"], log=None)
res, rows = rq1_metrics(succ, scores)
print(f"{len(branches)} branch states x K={P['K']} executed candidates ({time.time() - t0:.0f}s):", res)
M = pd.DataFrame(rows).set_index("scorer"); M

# %%
gate = M.get("auroc_fail_within", pd.Series(dtype=float)).drop("random", errors="ignore").max()
print(f"GO/NO-GO: best within-state AUROC = {gate:.3f} -> {'GO' if gate > 0.6 else 'NO-GO / pivot'}  (dnh week-1 criterion: > 0.6)")
fig, ax = plt.subplots(1, len(scores), figsize=(4 * len(scores), 3))
for a, (n, s) in zip(ax, scores.items()):
    a.hist(s[succ].ravel(), bins=20, alpha=.6, label="candidate succeeded", density=True)
    a.hist(s[~succ].ravel(), bins=20, alpha=.6, label="candidate failed", density=True)
    a.set_title(n); a.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3 · Closed-loop steering: base vs random-of-K vs best-of-K

# %%
METHODS = {"base (K=1)": dict(scorer=scorers[0], K=1), "random-of-K": dict(scorer=scorers[0], K=P["K"]),
           "best-of-K: WM σ": dict(scorer=s_alea, K=P["K"]), "best-of-K: contrast": dict(scorer=scorers[3], K=P["K"])}
t0 = time.time()
rows, paired = [], {}
for cond in ("nominal", "perturbed"):
    for i in range(P["n_eval"]):
        sc = sample_scenario(env.cfg, 730_000 + i, ood=[] if cond == "nominal" else [PERTURB[i % len(PERTURB)]])
        for name, m in METHODS.items():
            r = steered_episode(env, runner, tracker, m["scorer"], sc, K=m["K"], seed=i)
            rows.append(dict(condition=cond, method=name, success=r["success"], seed=i))
E = pd.DataFrame(rows)
print(f"{len(E)} episodes in {time.time() - t0:.0f}s")
E.pivot_table(index="method", columns="condition", values="success", aggfunc="mean").loc[list(METHODS)].round(2)

# %% [markdown]
# ## 4 · Abstention with a conformal threshold
# η is the (1-α) conformal quantile of the selected candidate's WM-σ score over nominal calibration
# episodes, so a nominal decision is flagged uncertified with probability ≤ α. With no certified
# candidate, the robot stops and hands over (counted as "abstained", not as success).

# %%
cal = [steered_episode(env, runner, tracker, s_alea, sample_scenario(env.cfg, 740_000 + i), K=P["K"], seed=i) for i in range(P["n_eval"])]
eta = conformal_threshold([max(r["scores"]) for r in cal if r["scores"]], alpha=0.1)
rows = []
for cond in ("nominal", "perturbed"):
    for i in range(P["n_eval"]):
        sc = sample_scenario(env.cfg, 750_000 + i, ood=[] if cond == "nominal" else [PERTURB[i % len(PERTURB)]])
        r = steered_episode(env, runner, tracker, s_alea, sc, K=P["K"], eta=eta, seed=i)
        rows.append(dict(condition=cond, outcome="abstained" if r["abstained"] else ("success" if r["success"] else "failure")))
A = pd.DataFrame(rows)
print(f"eta = {eta:.3f}")
pd.crosstab(A.condition, A.outcome, normalize="index").round(2)

# %% [markdown]
# ## 5 · Learn-then-Test: certify a steering strength λ for paired harm ≤ ε
# Paired episodes (same scenario and seed) with the base policy and with tilted steering at strength λ.
# Harm = base succeeded ∧ steered failed. Fixed-sequence LTT over increasing λ controls the
# family-wise error at δ. With few calibration episodes nothing can be certified at ε = 0.05
# (≥ 45 harm-free pairs are needed), which is exactly why this step belongs on the GPU nodes.

# %%
LAMS = [0.5, 2.0, np.inf]
base = [steered_episode(env, runner, tracker, s_alea, sample_scenario(env.cfg, 760_000 + i), K=1, seed=i)["success"] for i in range(P["n_eval"])]
harm, abst = {}, {}
for lam in LAMS:
    st = [steered_episode(env, runner, tracker, s_alea, sample_scenario(env.cfg, 760_000 + i), K=P["K"], lam=lam, seed=i) for i in range(P["n_eval"])]
    harm[lam] = [int(b and not s["success"]) for b, s in zip(base, st)]
    abst[lam] = [int(s["abstained"]) for s in st]
cal = ltt_calibrate(LAMS, harm, abst, eps=0.05, alpha=0.1, delta=0.1)
print(f"base success {np.mean(base):.2f}; certified: {cal['certified']} (lambda = {cal['lam']})")
pd.DataFrame(cal["table"])
