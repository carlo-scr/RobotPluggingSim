"""Inference-time steering of a frozen policy with a latent world model (mirrors the lab's `dnh` code).

At each replanning point: sample K candidate chunks from the policy, imagine each with the WM from the
current latent context, score it (lower = better), and execute the selected one. If no candidate's
score is below the abstention threshold eta, abstain (stop and hand over).
Also the RQ1 oracle ranking (execute every candidate from the restored sim state and let the base
policy finish) and Learn-then-Test calibration of the paired harm rate.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.stats import binom

from .wm import LatentWM, wm_features


# ---------------------------------------------------------------------- WM context during rollouts
class FeatureTracker:
    """Renders the WM cameras, encodes them and keeps the last H (x, action) pairs."""

    def __init__(self, encoder, cams, H=2, hw=(72, 128)):
        self.enc, self.cams, self.H, self.hw = encoder, list(cams), H, hw
        self.reset()

    def reset(self):
        self.xs, self.acts = [], []

    def observe(self, env):
        frames = np.stack([env.render(c, self.hw[1], self.hw[0]) for c in self.cams])
        z = self.enc.encode(frames).reshape(1, -1)
        x = wm_features(z, env.observe()[None])[0]
        self.xs.append(x)
        self.xs = self.xs[-self.H :]
        return x

    def record_action(self, a):
        self.acts.append(np.asarray(a, np.float32))
        self.acts = self.acts[-self.H :]

    def context(self):
        """hx [H,D] (left-padded), ha [H,A]: actions that led into each history state."""
        xs = [self.xs[0]] * (self.H - len(self.xs)) + self.xs
        prev = [np.zeros(16, np.float32)] + self.acts  # action into xs[-1] is acts[-1]
        ha = prev[-self.H :]
        ha = [ha[0]] * (self.H - len(ha)) + ha
        return np.stack(xs), np.stack(ha)

    def snapshot(self):
        return copy.deepcopy((self.xs, self.acts))

    def restore(self, snap):
        self.xs, self.acts = copy.deepcopy(snap)


# ---------------------------------------------------------------------- scorers (lower = better)
class RandomScorer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def score(self, hx, ha, chunks):
        return self.rng.random(len(chunks))


class WMScorer:
    """u(A) = max over the imagined chunk of the WM's predicted std (Ward et al.) or ensemble disagreement."""

    def __init__(self, wm: LatentWM, mode="alea", name=None, device="cpu"):
        self.wm, self.mode, self.device = wm, mode, device
        self.name = name or f"wm_{mode}"

    def per_step(self, hx, ha, chunks):
        K = len(chunks)
        t = lambda v: torch.as_tensor(np.asarray(v, np.float32), device=self.device)
        alea, epi = self.wm.imagine(t(hx)[None].repeat(K, 1, 1), t(ha)[None].repeat(K, 1, 1), t(chunks))
        return (alea if self.mode == "alea" else epi).cpu().numpy()

    def score(self, hx, ha, chunks):
        return self.per_step(hx, ha, chunks).max(-1)


class ContrastScorer:
    """c(A) = u_success(A) - beta * min(u_failure(A), kappa): nominal-like and unlike known failures."""

    name = "contrast"

    def __init__(self, s: WMScorer, f: WMScorer, beta=1.0, kappa=np.inf):
        self.s, self.f, self.beta, self.kappa = s, f, beta, kappa

    def score(self, hx, ha, chunks):
        return self.s.score(hx, ha, chunks) - self.beta * np.minimum(self.f.score(hx, ha, chunks), self.kappa)


def select(scores, lam=np.inf, eta=np.inf, rng=None):
    """Tilted selection (dnh.steering): lam=0 -> base policy (candidate 0), lam=inf -> best-of-K.
    Candidates with score > eta are not certified; none certified -> -1 (abstain)."""
    rng = rng or np.random.default_rng()
    ok = scores <= eta
    if not ok.any():
        return -1
    if lam == 0:
        return int(np.flatnonzero(ok)[0])
    z = (scores - scores.mean()) / (scores.std() + 1e-8)
    if np.isinf(lam):
        return int(np.argmin(np.where(ok, z, np.inf)))
    logits = np.where(ok, -lam * z, -np.inf)
    p = np.exp(logits - logits.max())
    return int(rng.choice(len(scores), p=p / p.sum()))


# ---------------------------------------------------------------------- closed-loop steering
def steered_episode(env, runner, tracker, scorer, scenario, K=8, lam=np.inf, eta=np.inf, max_s=16.0, settle_s=1.0, seed=0, record=()):
    """Run one episode with steering at every replan. Returns dict(success, abstained, decisions, ...)."""
    rng = np.random.default_rng(seed)
    env.reset(scenario)
    runner.reset()
    tracker.reset()
    frames = {c: [] for c in record}
    n_dec, t_latch, abst, chosen_scores = 0, None, False, []
    while env.time < max_s and not env.unstable:
        if len(runner.queue) < tracker.H or record:  # the WM context only needs the last H obs before a replan
            tracker.observe(env)
        runner.observe()
        for c in record:
            frames[c].append(env.render(c))
        if not runner.queue:
            chunks = runner.sample_chunks(K)
            if K == 1 and np.isinf(eta):
                k, sc = 0, np.zeros(1)
            else:
                hx, ha = tracker.context()
                sc = scorer.score(hx, ha, np.stack([runner.to_env_actions(c) for c in chunks]))
                k = select(sc, lam, eta, rng)
            n_dec += 1
            if k < 0:
                abst = True
                break
            chosen_scores.append(float(sc[k]))
            runner.set_plan(chunks[k])
        a = runner._env_action(runner.queue.pop(0))
        tracker.record_action(a)
        env.step(a)
        if env.latched_port >= 0 and t_latch is None:
            t_latch = env.time
        if t_latch is not None and env.time - t_latch > settle_s:
            break
    return dict(success=bool(env.latched_port == scenario.port_idx), abstained=abst, decisions=n_dec, latch_time=t_latch,
                duration=env.time, scores=chosen_scores, frames=frames)


# ---------------------------------------------------------------------- RQ1 oracle ranking
def capture_branches(env, runner, tracker, scenarios, at_replans=(2, 5, 8), max_s=16.0):
    """Run the base policy and capture full (sim, policy, WM-context) snapshots at given replan indices."""
    out = []
    for i, sc in enumerate(scenarios):
        r_at = at_replans[i % len(at_replans)]
        env.reset(sc)
        runner.reset()
        tracker.reset()
        n_plan = 0
        while env.time < max_s and not env.unstable and env.latched_port < 0:
            if len(runner.queue) < tracker.H:
                tracker.observe(env)
            runner.observe()
            if not runner.queue:
                if n_plan == r_at:
                    out.append(dict(scenario=sc, sim=env.get_state(), runner=copy.deepcopy((runner.hist_low, runner.hist_img)),
                                    tracker=tracker.snapshot(), replan=r_at))
                    break
                runner.set_plan(runner.sample_chunks(1)[0])
                n_plan += 1
            a = runner._env_action(runner.queue.pop(0))
            tracker.record_action(a)
            env.step(a)
    return out


def _restore(env, runner, tracker, b):
    env.reset(b["scenario"])
    env.set_state(b["sim"])
    runner.reset()
    runner.hist_low, runner.hist_img = copy.deepcopy(b["runner"])
    tracker.restore(b["tracker"])


def finish_with_base(env, runner, scenario, max_s=16.0, settle_s=1.0, observed=True):
    """Continue with the base policy. `observed`: the current observation is already in the runner's
    history (true right after restoring a branch), so the first queued action is executed directly."""
    t_latch = None
    first = observed and bool(runner.queue)
    while env.time < max_s and not env.unstable:
        if first:
            env.step(runner._env_action(runner.queue.pop(0)))
            first = False
        else:
            env.step(runner.act())
        if env.latched_port >= 0 and t_latch is None:
            t_latch = env.time
        if t_latch is not None and env.time - t_latch > settle_s:
            break
    return bool(env.latched_port == scenario.port_idx)


def rq1_ranking(env, runner, tracker, scorers, branches, K=8, max_s=16.0, log=print):
    """For each branch state: sample K chunks, score with every scorer, then execute each candidate
    from the restored state and let the base policy finish. -> (succ [N,K], scores {name: [N,K]})"""
    N = len(branches)
    succ = np.zeros((N, K), bool)
    scores = {s.name: np.zeros((N, K)) for s in scorers}
    for i, b in enumerate(branches):
        _restore(env, runner, tracker, b)
        chunks = runner.sample_chunks(K)
        env_chunks = np.stack([runner.to_env_actions(c) for c in chunks])
        hx, ha = tracker.context()
        for s in scorers:
            scores[s.name][i] = s.score(hx, ha, env_chunks)
        for k in range(K):
            _restore(env, runner, tracker, b)
            runner.set_plan(chunks[k])
            succ[i, k] = finish_with_base(env, runner, b["scenario"], max_s)
        if log:
            log(f"  branch {i + 1}/{N}: {succ[i].sum()}/{K} candidates succeed")
    return succ, scores


def rq1_metrics(succ, scores):
    from .monitor import auroc

    mixed = succ.any(1) & ~succ.all(1)
    res = dict(n_states=len(succ), base_success=float(succ.mean()), frac_mixed=float(mixed.mean()),
               oracle_success=float(succ.any(1).mean()))
    rows = []
    for n, s in scores.items():
        r = dict(scorer=n)
        if 0 < succ.mean() < 1:
            r["auroc_fail_pooled"] = auroc(s[succ], s[~succ])
        if mixed.any():
            best = s[mixed].argmin(1)
            r["p_success_best"] = float(succ[mixed][np.arange(mixed.sum()), best].mean())
            r["p_success_mean"] = float(succ[mixed].mean())
            r["steering_gain"] = r["p_success_best"] - r["p_success_mean"]
            r["auroc_fail_within"] = float(np.nanmean([auroc(s[i][succ[i]], s[i][~succ[i]]) for i in np.flatnonzero(mixed)]))
        rows.append(r)
    return res, rows


# ---------------------------------------------------------------------- Learn-then-Test (dnh.ltt)
def binom_pvalue(n_losses, n, level):
    return float(binom.cdf(n_losses, n, level))


def ltt_calibrate(lams, harm: dict, abst: dict, eps=0.05, alpha=0.1, delta=0.1):
    """Fixed-sequence LTT over increasing steering strength: certify the largest lam with
    paired-harm rate <= eps and false-abstention rate <= alpha (FWER <= delta)."""
    best, table = None, []
    for lam in lams:
        h, a = np.asarray(harm[lam]), np.asarray(abst[lam])
        p = max(binom_pvalue(int(h.sum()), len(h), eps), binom_pvalue(int(a.sum()), len(a), alpha))
        table.append(dict(lam=lam, harm=h.mean(), abst=a.mean(), p=p))
        if p > delta:
            break
        best = lam
    return dict(lam=0.0 if best is None else best, certified=best is not None, table=table)
