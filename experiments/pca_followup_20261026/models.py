"""Preserve EW7's fixed share and refit only the PCA classifier loss."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from experiments.neural_research_common.data import TARGET_ORDER
from experiments.pca_fisher_20261026.models import Bundle, LR_PARAMS, checked_matrix
from src.phase_b_rules import rank01

BASE_ENSEMBLES = {
    "h2_pca99_a025": {"H2": 0.75, "pca99_lr": 0.25},
    "h2_pca99_a050": {"H2": 0.50, "pca99_lr": 0.50},
    "h2_pca99_a075": {"H2": 0.25, "pca99_lr": 0.75},
    "h2_lr_a050": {"H2": 0.50, "lr": 0.50},
    "h2_fisher_a050": {"H2": 0.50, "fisher_lda": 0.50},
    "h2_pca99_fisher_equal": {"H2": 1 / 3, "pca99_lr": 1 / 3, "fisher_lda": 1 / 3},
}
WEIGHTED_RECIPES = {"weighted_pca99_c01": 0.1, "weighted_pca99_c1": 1.0,
                    "weighted_pca99_c10": 10.0}


def all_recipes():
    return {**BASE_ENSEMBLES, **{name: {name: 1.0} for name in WEIGHTED_RECIPES},
            **{f"h2_{name}_a050": {"H2": 0.5, name: 0.5} for name in WEIGHTED_RECIPES}}


def fixed_rank_scores(probabilities, ew7, model_weights):
    """Return model rank mixture and final score, without reranking the mixture."""
    if not model_weights or not np.isclose(sum(model_weights.values()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Model weights must sum to one")
    ew7 = np.asarray(ew7, dtype=np.float64)
    if ew7.ndim != 2 or ew7.shape[1] != len(TARGET_ORDER) or not len(ew7) or not np.isfinite(ew7).all():
        raise ValueError("Expected finite six-target EW7 scores")
    mixed = np.zeros_like(ew7)
    for name, weight in model_weights.items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Model weights must be finite and nonnegative")
        p = np.asarray(probabilities[name], dtype=np.float64)
        if p.shape != ew7.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Model probabilities must match EW7 and lie in [0,1]")
        mixed += weight * rank01(p)
    return mixed, 0.5 * mixed + 0.5 * rank01(ew7)


def fit_weighted_lr(name, C, preprocessor, x, y, weights, columns):
    x = checked_matrix(x, len(columns))
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    if tuple(columns) != preprocessor.columns or preprocessor.training_rows != len(x):
        raise ValueError("Preprocessing feature order or training row count mismatch")
    if y.shape != (len(x), len(TARGET_ORDER)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary targets")
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.mean(), 1):
        raise ValueError("Weights must be finite, positive, aligned and normalized to mean one")
    if C <= 0 or not np.isfinite(C):
        raise ValueError("Regularization C must be finite and positive")
    started = time.perf_counter()
    z = preprocessor.transform(x)
    models, fits = [], []
    for j, target in enumerate(TARGET_ORDER):
        target_started = time.perf_counter()
        values = np.unique(y[:, j])
        retry = False
        if len(values) == 1:
            model = int(values[0])
        else:
            model = LogisticRegression(**{**LR_PARAMS, "C": C})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(z, y[:, j], sample_weight=weights)
            except ConvergenceWarning:
                retry = True
                model.set_params(max_iter=4000)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(z, y[:, j], sample_weight=weights)
        models.append(model)
        fits.append({"target": target, "positive_count": int(y[:, j].sum()),
                     "constant_class": model if isinstance(model, int) else None,
                     "iterations": 0 if isinstance(model, int) else int(model.n_iter_.max()),
                     "convergence_retry_4000": retry, "fit_seconds": time.perf_counter() - target_started})
    return Bundle(name, preprocessor, tuple(TARGET_ORDER), tuple(models)), {
        "recipe": name, "C": C, "preprocessing_reused": True, "preprocessing_weighting": "uniform",
        "classifier_weighting": "H2 two-week half-life; normalized to mean one", "training_rows": len(x),
        "sample_weight_min": float(weights.min()), "sample_weight_max": float(weights.max()),
        "sample_weight_mean": float(weights.mean()), "dimensions": z.shape[1],
        "total_fit_seconds": time.perf_counter() - started, "target_fits": fits}
