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


def test_skill_score():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ref = y + 2.0
    assert abs(ev.skill_score(y, y, ref) - 1.0) < 1e-9   # perfect model -> skill 1
    assert ev.skill_score(y, ref, ref) == 0.0            # model == reference -> skill 0


def test_forward_chaining_no_leakage():
    splits = ev.forward_chaining_splits(120, n_splits=5)
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() < te.min()                 # train strictly precedes test
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
