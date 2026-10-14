"""Reusable prior-week heuristics, logistic, and histogram gradient boosting models.

No target labels or UserID/cutoff_utc are model input features.
Models fit on EXP-005 historical train rows only; validation is never used for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import TARGETS

MODELS = ("zero", "persistence", "logistic", "hgb")


def feature_names(frame: pd.DataFrame) -> list[str]:
    forbidden = {"UserID", "cutoff_utc", *TARGETS}
    bad = forbidden & set(frame.columns) - {"UserID", "cutoff_utc"}
    if bad:
        raise ValueError(f"Label leakage: {sorted(bad)} in features")
    columns = [c for c in frame.columns if c not in forbidden]
    if not columns:
        raise ValueError("No feature columns")
    for col in columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"Non-numeric model feature {col!r}")
    if frame[columns].isna().any().any():
        raise ValueError("Missing values in features: EXP-005 promised complete numeric data")
    return columns


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing inference/validation features: {sorted(missing)}")
    if (set(TARGETS) & set(frame.columns)):
        raise ValueError("Target labels unexpectedly present in feature input")
    x = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values")
    return x


@dataclass
class ModelBundle:
    kind: str
    feature_columns: list[str]
    classifiers: dict[str, Any]
    constant_labels: dict[str, int]
    thresholds: dict[str, float]
    # Assigned after calibration. Serialized only from trusted local training.


def fit_bundle(kind: str, features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> ModelBundle:
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}")
    columns = feature_names(features)
    if kind in ("zero", "persistence"):
        for target in TARGETS:
            if kind == "persistence" and f"{target}_active_prior_7d" not in columns:
                raise ValueError(f"Persistence flag missing for {target}")
        return ModelBundle(kind, columns, {}, {}, {t: 0.5 for t in TARGETS})
    x = feature_matrix(features, columns)
    models = {}
    constants = {}
    for target in TARGETS:
        y = labels[target].to_numpy(dtype=np.int8)
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"Invalid labels for {target}")
        if len(np.unique(y)) == 1:
            constants[target] = int(y[0])
            # A constant model is more honest than inventing a positive class.
            models[target] = DummyClassifier(strategy="constant", constant=int(y[0])).fit(x, y)
        elif kind == "logistic":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(cfg["logistic_C"]), max_iter=int(cfg["logistic_max_iter"]),
                    solver="lbfgs", class_weight=None, random_state=int(cfg["random_state"]),
                ),
            )
            model.fit(x, y)
            models[target] = model
        else:
            model = HistGradientBoostingClassifier(
                learning_rate=float(cfg["hgb_learning_rate"]),
                max_iter=int(cfg["hgb_max_iter"]),
                max_leaf_nodes=int(cfg["hgb_max_leaf_nodes"]),
                min_samples_leaf=int(cfg["hgb_min_samples_leaf"]),
                l2_regularization=float(cfg["hgb_l2_regularization"]),
                early_stopping=False,  # no random holdout across weeks
                random_state=int(cfg["random_state"]),
            )
            model.fit(x, y)
            models[target] = model
    return ModelBundle(kind, columns, models, constants, {t: 0.5 for t in TARGETS})


def probabilities(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    if bundle.kind == "zero":
        return np.zeros((len(features), len(TARGETS)), dtype=np.float64)
    if bundle.kind == "persistence":
        return features[[f"{t}_active_prior_7d" for t in TARGETS]].to_numpy(dtype=np.float64)
    x = feature_matrix(features, bundle.feature_columns)
    result = np.empty((len(x), len(TARGETS)), dtype=np.float64)
    for j, target in enumerate(TARGETS):
        if target in bundle.constant_labels:
            result[:, j] = float(bundle.constant_labels[target])
        else:
            model = bundle.classifiers[target]
            idx = list(model.classes_ if bundle.kind == "hgb" else model.named_steps["logisticregression"].classes_).index(1)
            result[:, j] = model.predict_proba(x)[:, idx]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model probabilities")
    return result
