"""Conformal failure monitoring on top of the latent WM (Ward et al.; three-way decision as in Ho et al.).

Per episode: a running non-conformity score s_t (WM sigma or one-step NLL, smoothed), its trajectory
max S = max_t s_t, a split-conformal threshold tau = the ceil((n+1)(1-alpha))-th smallest S over
nominal calibration episodes (so P(false alarm on a nominal episode) <= alpha), and the first time
s_t > tau as the detection time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth(x, k=3):
    if k <= 1 or len(x) < k:
        return np.asarray(x)
    return np.convolve(np.asarray(x, float), np.ones(k) / k, mode="same")


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


def evaluate_monitor(traces: dict, anns: dict, splits: dict, key="alea", alpha=0.1, k_smooth=3, dt=0.05):
    """traces[name][episode_id] -> dict of per-step arrays; anns[episode_id] -> annotations;
    splits: name -> list of episode ids with 'nominal_cal' used for calibration.
    Returns (per-episode DataFrame, threshold)."""
    S = {e: smooth(traces[e][key], k_smooth) for e in traces}
    tau = conformal_threshold([S[e].max() for e in splits["nominal_cal"]], alpha)
    rows = []
    for split, ids in splits.items():
        if split == "nominal_cal":
            continue
        for e in ids:
            a = anns[e]
            tdet = first_crossing(S[e], tau)
            tev = event_time(a)
            rows.append(dict(episode=e, split=split, category=a["category"], outcome=a["outcome"], failure_mode=a["failure_mode"],
                             injected=(a["injected_failure"] or {}).get("kind"), ood="+".join(a["ood"]), score=float(S[e].max()),
                             flagged=tdet is not None, t_detect=None if tdet is None else tdet * dt, t_event=tev,
                             lead_s=None if (tdet is None or tev is None) else tev - tdet * dt))
    return pd.DataFrame(rows), tau


def summarize(df: pd.DataFrame):
    """Alarm rates by group + AUROC of the episode score against nominal-test episodes."""
    nom = df[df.category == "nominal"]
    out = []
    for name, g in [("nominal (FPR)", nom)] + [(f"failure: {k}", v) for k, v in df[df.category == "known_failure"].groupby("injected")] + \
            [(f"ood: {k}", v) for k, v in df[df.category == "ood"].groupby("ood")]:
        out.append(dict(group=name, n=len(g), alarm_rate=g.flagged.mean(), auroc_vs_nominal=np.nan if g is nom else auroc(nom.score, g.score),
                        physical_failures=(g.outcome == "failure").mean(), median_lead_s=g.lead_s.dropna().median() if len(g.lead_s.dropna()) else np.nan))
    return pd.DataFrame(out)


def three_way(score_s, score_f, tau_s, tau_f):
    """Ho et al. decision: nominal if the success-WM is not surprised; else known failure if the
    failure-WM explains it; else OOD."""
    if score_s <= tau_s:
        return "nominal"
    return "known_failure" if score_f <= tau_f else "ood"
