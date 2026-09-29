import numpy as np

from single_photon_mei.stage2 import greedy_select_diverse_targets
from single_photon_mei.utils.encoding import tuning_correlation


def test_greedy_selection_rejects_threshold_and_accepts_negative_tuning():
    base = np.linspace(-1, 1, 20)
    responses = np.column_stack([base, base, -base, np.sin(np.arange(20))])
    selected, audit = greedy_select_diverse_targets(
        responses,
        np.arange(4),
        np.array([0.9, 0.8, 0.7, 0.6]),
        threshold=0.8,
        target_count=3,
    )
    assert selected.tolist() == [0, 2, 3]
    assert not bool(audit.loc[audit.unit_id_zero_based == 1, "accepted"].iloc[0])


def test_greedy_order_breaks_score_ties_by_unit_id():
    rng = np.random.default_rng(2)
    responses = rng.normal(size=(30, 3))
    selected, _ = greedy_select_diverse_targets(
        responses, np.array([2, 0, 1]), np.ones(3), threshold=1.0, target_count=3
    )
    assert selected.tolist() == [0, 1, 2]


def test_saved_target_order_defines_rdm_order():
    responses = np.array([[0, 0, 2], [1, -1, 1], [2, -2, 0]], dtype=float)
    selected = np.array([2, 0])
    correlation = tuning_correlation(responses[:, selected])
    assert correlation.shape == (2, 2)
    np.testing.assert_allclose(correlation[0, 1], -1.0)
