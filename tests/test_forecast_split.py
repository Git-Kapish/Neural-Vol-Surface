import numpy as np

from nvs.nn.forecaster import make_xy, project_calendar, walk_forward_splits


def test_features_strictly_precede_targets():
    F = np.arange(40.0).reshape(-1, 1) * np.ones((1, 2))  # F[t] == t, so values reveal time
    X, Y, t = make_xy(F, lags=3)
    assert (X[:, :3 * 2].reshape(len(X), 3, 2)[:, :, 0].max(axis=1) <= t).all()  # latest feature = t
    assert (Y[:, 0] == t + 1).all()  # target is the next day, strictly after every feature


def test_walk_forward_is_time_ordered_and_covers_test_once():
    n, seen = 300, []
    for r, tgt in walk_forward_splits(n, min_train=100, step=40):
        assert tgt.min() >= r  # all test targets are after the training window [0, r)
        assert np.all(np.diff(tgt) == 1)
        seen += tgt.tolist()
    assert seen == list(range(100, n))  # contiguous, no shuffling, no overlap


def test_calendar_projection_removes_violations():
    T = np.array([0.1, 0.5, 1.0])
    iv = np.array([[0.30, 0.30], [0.20, 0.20], [0.25, 0.25]])  # w drops from T=0.1 to 0.5
    w = project_calendar(iv, T) ** 2 * T[:, None]
    assert (np.diff(w, axis=0) >= -1e-15).all()
