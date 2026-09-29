from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def pearson_by_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim not in (1, 2):
        raise ValueError("y_true and y_pred must have matching sample-first shapes")
    one_dimensional = y_true.ndim == 1
    if one_dimensional:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    true_centered = y_true - y_true.mean(axis=0)
    pred_centered = y_pred - y_pred.mean(axis=0)
    denominator = np.sqrt(np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    scores = np.zeros(y_true.shape[1], dtype=np.float64)
    np.divide(np.sum(true_centered * pred_centered, axis=0), denominator, out=scores, where=denominator > eps)
    return scores[0] if one_dimensional else scores


def zscore_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    safe = np.where(std > np.finfo(np.float32).eps, std, 1.0)
    return ((values - mean) / safe).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _fit_preprocessor(train: np.ndarray, test: np.ndarray, components: int | None, random_state: int):
    if components is None:
        return train, test
    pca = PCA(n_components=components, random_state=random_state)
    return pca.fit_transform(train), pca.transform(test)


@dataclass
class NestedRidgeResult:
    predictions: np.ndarray
    scores: np.ndarray
    selected_alphas: np.ndarray
    fold_indices: list[tuple[np.ndarray, np.ndarray]]


def nested_ridge_cv(
    features: np.ndarray,
    responses: np.ndarray,
    *,
    alphas: list[float],
    outer_folds: int,
    inner_folds: int,
    pca_components: int | None,
    random_state: int,
) -> NestedRidgeResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    outer = KFold(outer_folds, shuffle=True, random_state=random_state)
    predictions = np.empty_like(y)
    selected = []
    fold_indices = []
    for train, test in outer.split(x):
        inner = KFold(inner_folds, shuffle=True, random_state=random_state)
        inner_predictions = {float(alpha): np.empty_like(y[train]) for alpha in alphas}
        for inner_train, inner_val in inner.split(train):
            train_global = train[inner_train]
            val_global = train[inner_val]
            x_train, x_val = _fit_preprocessor(
                x[train_global], x[val_global], pca_components, random_state
            )
            for alpha in alphas:
                model = Ridge(alpha=float(alpha), fit_intercept=True)
                model.fit(x_train, y[train_global])
                inner_predictions[float(alpha)][inner_val] = np.asarray(model.predict(x_val)).reshape(
                    len(inner_val), y.shape[1]
                )
        best_alpha = max(
            (float(alpha) for alpha in alphas),
            key=lambda alpha: float(np.mean(pearson_by_target(y[train], inner_predictions[alpha]))),
        )
        x_train, x_test = _fit_preprocessor(x[train], x[test], pca_components, random_state)
        model = Ridge(alpha=best_alpha, fit_intercept=True)
        model.fit(x_train, y[train])
        predictions[test] = np.asarray(model.predict(x_test)).reshape(len(test), y.shape[1])
        selected.append(best_alpha)
        fold_indices.append((train.copy(), test.copy()))
    return NestedRidgeResult(
        predictions=predictions,
        scores=np.asarray(pearson_by_target(y, predictions)),
        selected_alphas=np.asarray(selected),
        fold_indices=fold_indices,
    )


@dataclass
class ReadoutResult:
    weight: np.ndarray
    bias: np.ndarray
    selected_alphas: np.ndarray
    cv_scores: np.ndarray
    predictions: np.ndarray


def fit_per_target_readout(
    features: np.ndarray,
    target_responses: np.ndarray,
    *,
    alphas: list[float],
    cv_folds: int,
    pca_components: int | None,
    random_state: int,
) -> ReadoutResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target_responses, dtype=np.float64)
    alpha_predictions = {float(alpha): np.empty_like(y) for alpha in alphas}
    cv = KFold(cv_folds, shuffle=True, random_state=random_state)
    splits = [(train.copy(), test.copy()) for train, test in cv.split(x)]
    for train, test in splits:
        x_train, x_test = _fit_preprocessor(x[train], x[test], pca_components, random_state)
        for alpha in alphas:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(x_train, y[train])
            alpha_predictions[float(alpha)][test] = np.asarray(model.predict(x_test)).reshape(
                len(test), y.shape[1]
            )
    alpha_scores = np.vstack([pearson_by_target(y, alpha_predictions[float(alpha)]) for alpha in alphas])
    best_indices = np.argmax(alpha_scores, axis=0)
    selected = np.asarray([float(alphas[index]) for index in best_indices])
    scores = alpha_scores[best_indices, np.arange(y.shape[1])]

    full_pca = PCA(n_components=pca_components, random_state=random_state) if pca_components is not None else None
    transformed = full_pca.fit_transform(x) if full_pca is not None else x
    weights = np.empty((y.shape[1], x.shape[1]), dtype=np.float64)
    biases = np.empty(y.shape[1], dtype=np.float64)
    for target, alpha in enumerate(selected):
        ridge = Ridge(alpha=float(alpha), fit_intercept=True).fit(transformed, y[:, target])
        weight = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
        bias = float(ridge.intercept_)
        if full_pca is not None:
            weight = weight @ np.asarray(full_pca.components_, dtype=np.float64)
            bias -= float(np.asarray(full_pca.mean_) @ weight)
        weights[target] = weight
        biases[target] = bias
    weight = weights
    bias = biases
    full_predictions = x @ weight.T + bias
    return ReadoutResult(weight, bias, selected, np.asarray(scores), full_predictions)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def permutation_pvalues(
    responses: np.ndarray,
    predictions: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
    device: str = "cpu",
    unit_block: int = 256,
    permutation_block: int = 20,
) -> np.ndarray:
    y, _, y_std = zscore_columns(responses)
    p, _, p_std = zscore_columns(predictions)
    valid = (y_std > np.finfo(np.float32).eps) & (p_std > np.finfo(np.float32).eps)
    observed = np.mean(y * p, axis=0)
    exceed = np.zeros(y.shape[1], dtype=np.int64)
    rng = np.random.default_rng(random_state)
    permutations = np.stack([rng.permutation(y.shape[0]) for _ in range(n_permutations)])
    torch_device = torch.device(device)
    for start in range(0, y.shape[1], unit_block):
        stop = min(start + unit_block, y.shape[1])
        if not valid[start:stop].any():
            continue
        y_tensor = torch.from_numpy(y[:, start:stop]).to(torch_device)
        p_tensor = torch.from_numpy(p[:, start:stop]).to(torch_device)
        observed_tensor = torch.from_numpy(observed[start:stop]).abs().to(torch_device)
        counts = torch.zeros(stop - start, dtype=torch.int64, device=torch_device)
        for permutation_start in range(0, n_permutations, permutation_block):
            indices = torch.from_numpy(
                permutations[permutation_start : permutation_start + permutation_block]
            ).to(torch_device)
            null = torch.mean(y_tensor[indices] * p_tensor.unsqueeze(0), dim=1).abs()
            counts += torch.sum(null >= observed_tensor.unsqueeze(0), dim=0)
        exceed[start:stop] = counts.cpu().numpy()
    result = (exceed + 1) / (n_permutations + 1)
    result[~valid] = 1.0
    return result


def tuning_correlation(responses: np.ndarray) -> np.ndarray:
    normalized, _, _ = zscore_columns(responses)
    correlation = normalized.T @ normalized / normalized.shape[0]
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)
