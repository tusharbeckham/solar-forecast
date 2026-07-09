import numpy as np

from solar_forecast import evaluate as ev


def test_metrics_known_values():
    y = np.array([0.0, 2.0, 4.0])
    assert ev.rmse(y, y) == 0.0
    assert ev.mae(y, y) == 0.0
    off = y + 1.0
    assert abs(ev.mae(y, off) - 1.0) < 1e-9
    assert abs(ev.rmse(y, off) - 1.0) < 1e-9
    assert abs(ev.mbe(y, off) - 1.0) < 1e-9


def test_r2_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(ev.r2(y, y) - 1.0) < 1e-9                     # perfect fit
    assert abs(ev.r2(y, np.full_like(y, y.mean()))) < 1e-9   # mean predictor -> 0


def test_skill_score():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ref = y + 2.0
    assert abs(ev.skill_score(y, y, ref) - 1.0) < 1e-9   # perfect model -> skill 1
    assert ev.skill_score(y, ref, ref) == 0.0            # model == reference -> skill 0


def test_metrics_table_has_all_keys():
    y = np.array([1.0, 2.0, 3.0])
    yhat = y + 0.5
    clear = y + 1.0
    persist = y + 2.0
    table = ev.metrics_table(y, yhat, clear, persist)
    for key in ("mae", "rmse", "mbe", "r2", "skill_vs_clear_sky", "skill_vs_persistence"):
        assert key in table


def test_forward_chaining_no_leakage():
    splits = ev.forward_chaining_splits(120, n_splits=5)
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() < te.min()                 # train strictly precedes test
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
