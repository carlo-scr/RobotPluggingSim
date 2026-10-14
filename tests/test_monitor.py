"""Monitor maths on synthetic data: smoothing, failure times, OOD families, the conformal false-alarm
guarantee (also for combined scores), kNN novelty, and WM fine-tuning / early stopping."""
import numpy as np
import pytest

from plugsim.learn.monitor import KNNNovelty, evaluate_monitor, failure_time, headline, ood_family, smooth, summarize
from plugsim.learn.wm import score_episode, train_wm


def test_smooth_has_no_edge_dip():
    assert np.allclose(smooth(np.full(10, 5.0), 3), 5.0)


def test_failure_time():
    ph = [["approach", 0.0], ["lift", 2.0], ["insert", 4.0], ["release", 6.0], ["insert", 7.0], ["release", 8.0]]
    assert failure_time(dict(failure_mode="dropped", drop_times_s=[3.1, 5.0], phase_events=ph)) == 3.1
    assert failure_time(dict(failure_mode="grasp_missed", phase_events=ph)) == 2.0
    assert failure_time(dict(failure_mode="not_seated", phase_events=ph)) == 6.0  # the first release
    assert failure_time(dict(failure_mode="wrong_port", latch_time_s=5.5, phase_events=ph)) == 5.5
    assert failure_time(dict(failure_mode="none", phase_events=ph)) is None


def test_ood_family():
    assert ood_family(["lighting_dim"]) == "lighting"
    assert ood_family(["lighting_dim", "lighting_harsh"]) == "lighting"
    assert ood_family(["lighting_dim", "camera_noise"]) == "mixed"
    assert ood_family([]) == ""


def _synthetic(n_nom=400, n_fail=60, seed=0):
    rng = np.random.default_rng(seed)
    traces, anns, splits = {}, {}, {"nominal_cal": [], "nominal_test": [], "known_failure_test": []}
    to_cal = set(rng.permutation(n_nom)[: n_nom // 2])  # random split, as split_by_category does
    for i in range(n_nom + n_fail):
        fail = i >= n_nom
        tr = dict(a=rng.normal(0, 1, 100), b=rng.gamma(2, 1, 100))
        if fail:
            tr["a"][60:] += 6.0  # a clear jump at step 60, failure complete at 4 s (step 80)
        e = f"ep{i}"
        traces[e] = tr
        anns[e] = dict(category="known_failure" if fail else "nominal", outcome="failure" if fail else "success",
                       failure_mode="dropped" if fail else "none", drop_times_s=[4.0] if fail else [], injected_failure={"kind": "drop"} if fail else None,
                       injection_time_s=3.0 if fail else None, ood=[], phase_events=[])
        splits["known_failure_test" if fail else ("nominal_cal" if i in to_cal else "nominal_test")].append(e)
    return traces, anns, splits


@pytest.mark.parametrize("key", ["a", ("a", "b")])
def test_conformal_false_alarms_and_detection(key):
    # the guarantee is marginal over calibration draws (a single split can be off by chance), so average seeds
    fpr = []
    for seed in range(8):
        traces, anns, splits = _synthetic(seed=seed)
        df, norm = evaluate_monitor(traces, anns, splits, key=key, alpha=0.1)
        h = headline(df)
        fpr.append(h["nominal_FPR"])
        assert h["failure_TPR"] > 0.95 and h["failure_AUROC"] > 0.99
        assert h["warned_before_failure"] > 0.95  # detected around step 60, failure complete at step 80
    assert np.mean(fpr) <= 0.1 + 0.02
    assert set(norm) == ({key} if isinstance(key, str) else set(key))
    assert "failure: drop" in summarize(df).index


def test_knn_novelty_ranks_shifted_states_higher():
    rng = np.random.default_rng(0)
    train = [rng.normal(0, 1, (200, 8)) for _ in range(5)]
    knn = KNNNovelty(train, k=5)
    assert knn.score(rng.normal(0, 1, (50, 8))).mean() < knn.score(rng.normal(3, 1, (50, 8))).mean()


def test_wm_early_stopping_and_finetune():
    rng = np.random.default_rng(0)
    mk = lambda n: [dict(x=np.cumsum(rng.normal(0, 0.1, (40, 6)), 0).astype(np.float32), action=rng.normal(0, 1, (40, 3)).astype(np.float32)) for _ in range(n)]
    wm, hist = train_wm(mk(8), val=mk(2), steps=60, bs=64, hidden=32, n_members=2, log=None, log_every=20)
    assert [s for s, _ in wm.val_hist] == [0, 20, 40, 59] and len(hist) == 60
    ft, _ = train_wm(mk(4), init=wm, steps=10, bs=64, log=None)
    assert len(ft.members) == 2 and float((ft.d_std - wm.d_std).abs().max()) == 0.0  # normalisation kept
    assert score_episode(ft, mk(1)[0]["x"], mk(1)[0]["action"])["nll"].shape == (39,)
