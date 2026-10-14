"""Conformal failure monitoring on top of the latent WM (Ward et al.; three-way decision as in Ho et al.).

Per episode: a running non-conformity score s_t (WM sigma, one-step NLL or state novelty, smoothed),
its trajectory max S = max_t s_t, a split-conformal threshold tau = the ceil((n+1)(1-alpha))-th
smallest S over nominal calibration episodes (so P(false alarm on a nominal episode) <= alpha), and
the first time s_t > tau as the detection time. Several scores are combined with a Bonferroni split
(alarm when any score crosses its alpha/k threshold), which keeps the same guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..scenario import OOD_GROUPS


def smooth(x, k=3):
    """Centred moving average (the window shrinks at the ends instead of padding with zeros)."""
    x = np.asarray(x, float)
    if k <= 1 or len(x) < k:
        return x
    w = np.ones(k)
    return np.convolve(x, w, mode="same") / np.convolve(np.ones_like(x), w, mode="same")


def conformal_threshold(cal_scores, alpha=0.1):
    s = np.sort(np.asarray(cal_scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")


def first_crossing(trace, tau):
    i = np.flatnonzero(np.asarray(trace) > tau)
    return int(i[0]) if len(i) else None


def auroc(neg, pos):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = pd.Series(allv).rank().values
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def event_time(ann):
    """Ground-truth time at which the failure becomes physically evident (or the injection time)."""
    if ann.get("drop_times_s"):
        return ann["drop_times_s"][0]
    return ann.get("injection_time_s")


def failure_time(ann):
    """When a physical failure is complete (None for successes): the drop, the lift without the plug,
    the release without a latch, or the latch in the wrong port. Alarms before it are early warnings."""
    first = {}
    for ph, t in ann.get("phase_events") or []:
        first.setdefault(ph, t)
    mode = ann.get("failure_mode")
    if mode == "dropped" and ann.get("drop_times_s"):
        return ann["drop_times_s"][0]
    if mode == "grasp_missed":
        return first.get("lift")
    if mode == "not_seated":
        return first.get("release")
    if mode == "wrong_port":
        return ann.get("latch_time_s")
    return None


def ood_family(shifts):
    """['lighting_dim', 'camera_noise'] -> 'mixed'; one family -> its name (see scenario.OOD_GROUPS)."""
    fam = {g for s in shifts for g, members in OOD_GROUPS.items() if s in members}
    return fam.pop() if len(fam) == 1 else ("mixed" if fam else "")


def calibrate(traces: dict, cal_ids, key="nll", alpha=0.1, k_smooth=3):
    """Conformal calibration of one score or a tuple of scores (Bonferroni: each gets alpha / len(key))
    on nominal episodes -> {key: (median, tau)} of the smoothed episode maxima."""
    keys = (key,) if isinstance(key, str) else tuple(key)
    norm = {}
    for k in keys:
        cal = [smooth(traces[e][k], k_smooth).max() for e in cal_ids]
        norm[k] = (float(np.median(cal)), conformal_threshold(cal, alpha / len(keys)))
        if not np.isfinite(norm[k][1]):
            print(f"warning: {len(cal)} calibration episodes are too few for level {alpha / len(keys):.3f} "
                  f"(need {int(np.ceil(len(keys) / alpha)) - 1}); '{k}' will never alarm")
    return norm


def combined_trace(trace: dict, norm: dict, k_smooth=3):
    """Per-step score normalised as (s - median) / (tau - median) and maxed over the calibrated keys:
    > 1 means at least one score is past its threshold, and the scale is comparable across keys."""
    return np.max([(smooth(trace[k], k_smooth) - m) / max(tau - m, 1e-12) for k, (m, tau) in norm.items()], 0)


def evaluate_monitor(traces: dict, anns: dict, splits: dict, key="nll", alpha=0.1, k_smooth=3, dt=0.05):
    """traces[episode_id] -> dict of per-step arrays; anns[episode_id] -> annotations;
    splits: name -> list of episode ids with 'nominal_cal' used for calibration; key: a score or a tuple.
    Returns (per-episode DataFrame with the normalised episode `score` (> 1 = alarm), calibrate() output)."""
    norm = calibrate(traces, splits["nominal_cal"], key, alpha, k_smooth)
    rows = []
    for split, ids in splits.items():
        if split == "nominal_cal":
            continue
        for e in ids:
            a = anns[e]
            c = combined_trace(traces[e], norm, k_smooth)
            tdet = first_crossing(c, 1.0)
            tfail = failure_time(a) if a["outcome"] == "failure" else None
            rows.append(dict(episode=e, split=split, category=a["category"], outcome=a["outcome"], failure_mode=a["failure_mode"],
                             injected=(a["injected_failure"] or {}).get("kind"), ood="+".join(a["ood"]), ood_family=ood_family(a["ood"]),
                             score=float(c.max()), flagged=tdet is not None, t_detect=None if tdet is None else tdet * dt,
                             t_onset=event_time(a), t_fail=tfail, lead_s=None if (tdet is None or tfail is None) else tfail - tdet * dt))
    return pd.DataFrame(rows), norm


def _group_row(name, g, nom_ok, pos):
    fail, ok = g[g.outcome == "failure"], g[g.outcome == "success"]
    lead = fail.lead_s.dropna()
    return dict(group=name, n=len(g), physical_failures=len(fail) / max(len(g), 1),
                alarm_on_failures=fail.flagged.mean() if len(fail) else np.nan,
                alarm_on_successes=ok.flagged.mean() if len(ok) else np.nan,
                auroc=auroc(nom_ok.score, pos.score),
                median_lead_s=lead.median() if len(lead) else np.nan,
                warned_before_failure=(lead > 0).sum() / len(fail) if len(fail) else np.nan)


def summarize(df: pd.DataFrame):
    """Detection by group, separating the episodes that physically failed from those that did not.

    alarm_on_failures      TPR on the physically failed episodes of the group
    alarm_on_successes     alarm rate on the ones that still succeeded (for nominal: the false-alarm rate)
    auroc                  episode score vs successful nominal-test episodes, for the group's failures
                           (nominal: the few nominal episodes that failed anyway; OOD: every shifted episode)
    warned_before_failure  share of the group's failures detected before the failure was complete"""
    nom = df[df.category == "nominal"]
    nom_ok = nom[nom.outcome == "success"]
    failed = lambda g: g[g.outcome == "failure"]
    out = [_group_row("nominal", nom, nom_ok, failed(nom))]
    out += [_group_row(f"failure: {k}", g, nom_ok, failed(g)) for k, g in df[df.category == "known_failure"].groupby("injected")]
    out += [_group_row(f"ood: {k}", g, nom_ok, g) for k, g in df[df.category == "ood"].groupby("ood_family")]
    return pd.DataFrame(out).set_index("group")


def headline(df: pd.DataFrame):
    """One row: false alarms, detection of physical failures (AUROC, TPR, lead), alarms on injected
    failures that recovered, and OOD detection."""
    nom = df[(df.category == "nominal") & (df.outcome == "success")]
    kf = df[df.category == "known_failure"]
    fail, rec = kf[kf.outcome == "failure"], kf[kf.outcome == "success"]
    ood = df[df.category == "ood"]
    lead = fail.lead_s.dropna()
    return dict(nominal_FPR=nom.flagged.mean(), failure_AUROC=auroc(nom.score, fail.score), failure_TPR=fail.flagged.mean(),
                warned_before_failure=(lead > 0).sum() / max(len(fail), 1), median_lead_s=lead.median() if len(lead) else np.nan,
                recovered_alarm_rate=rec.flagged.mean(), OOD_AUROC=auroc(nom.score, ood.score), OOD_TPR=ood.flagged.mean())


def window_mean(x, i, w):
    """Mean of x over [i - w, i + w] (clipped to the array)."""
    return float(np.mean(x[max(0, i - w): i + w + 1]))


def three_way(score_s, score_f, tau_s, tau_f):
    """Ho et al. decision: nominal if the success-WM is not surprised; else known failure if the
    failure-WM explains it; else OOD."""
    if score_s <= tau_s:
        return "nominal"
    return "known_failure" if score_f <= tau_f else "ood"


class KNNNovelty:
    """State novelty: mean distance from x_t to its k nearest nominal training states (z-scored), the
    deep-kNN OOD score (Sun et al. 2022). It catches shifts that leave the dynamics alone: a new cable
    colour or dim lighting moves the latents but barely their deltas, so the one-step NLL stays nominal."""

    def __init__(self, train_x, k=5, stride=2, max_ref=50000, device="cpu", seed=0):
        X = np.concatenate([x[::stride] for x in train_x]).astype(np.float32)
        if len(X) > max_ref:
            X = X[np.random.default_rng(seed).choice(len(X), max_ref, replace=False)]
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        self.k, self.device = k, device
        self.ref = torch.as_tensor((X - self.mean) / self.std, device=device)

    @torch.no_grad()
    def score(self, x, bs=1024):
        """x [T, D] -> [T] (divided by sqrt(D), so roughly a per-dimension distance)."""
        q = torch.as_tensor((np.asarray(x, np.float32) - self.mean) / self.std, device=self.device)
        out = [torch.cdist(q[i: i + bs], self.ref).topk(self.k, largest=False).values.mean(1).cpu() for i in range(0, len(q), bs)]
        return torch.cat(out).numpy() / np.sqrt(q.shape[1])
