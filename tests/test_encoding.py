import numpy as np

from single_photon_mei.utils.encoding import fit_per_target_readout, nested_ridge_cv, pearson_by_target


def test_pearson_and_exported_readout_are_consistent():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 8))
    y = x @ rng.normal(size=(8, 3)) + rng.normal(scale=0.05, size=(40, 3))
    result = fit_per_target_readout(x, y, alphas=[0.01, 1, 100], cv_folds=4, pca_components=5, random_state=42)
    np.testing.assert_allclose(result.predictions, x @ result.weight.T + result.bias, rtol=1e-11, atol=1e-11)
    assert np.all(pearson_by_target(y, result.predictions) > 0.5)


def test_nested_cv_is_deterministic():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 7))
    y = rng.normal(size=(30, 4))
    kwargs = dict(alphas=[0.1, 1], outer_folds=3, inner_folds=2, pca_components=4, random_state=42)
    left = nested_ridge_cv(x, y, **kwargs)
    right = nested_ridge_cv(x, y, **kwargs)
    np.testing.assert_array_equal(left.selected_alphas, right.selected_alphas)
    np.testing.assert_allclose(left.predictions, right.predictions)


def test_single_target_readout_keeps_two_dimensional_shapes():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 6))
    y = rng.normal(size=(24, 1))
    result = fit_per_target_readout(x, y, alphas=[0.1, 1], cv_folds=3, pca_components=4, random_state=42)
    assert result.weight.shape == (1, 6)
    assert result.predictions.shape == (24, 1)
