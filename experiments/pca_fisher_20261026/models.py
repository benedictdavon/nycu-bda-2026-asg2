"""Training-only preprocessing and six independently fitted binary classifiers."""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.neural_research_common.data import BINARY_COLUMNS, TARGET_ORDER

RECIPES = {"lr": None, "pca90_lr": 0.90, "pca95_lr": 0.95,
           "pca99_lr": 0.99, "fisher_lda": None}
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-4,
                 class_weight=None, random_state=42)
LDA_PARAMS = dict(solver="lsqr", shrinkage="auto", priors=None)


def checked_matrix(values, width):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != width or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite matrix in the registered feature order")
    return x


class Preprocessor:
    def __init__(self, columns, retained_variance=None):
        self.columns = tuple(columns)
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Feature columns must be nonempty and unique")
        if {"UserID", "cutoff_utc", *TARGET_ORDER}.intersection(self.columns):
            raise ValueError("Identity or target columns cannot enter preprocessing")
        self.continuous = np.array([c not in BINARY_COLUMNS for c in self.columns])
        self.retained_variance = retained_variance

    def _logged(self, values):
        x = checked_matrix(values, len(self.columns)).copy()
        if not np.isin(x[:, ~self.continuous], [0, 1]).all():
            raise ValueError("Registered binary columns must contain only 0/1")
        c = x[:, self.continuous]
        x[:, self.continuous] = np.sign(c) * np.log1p(np.abs(c))
        return x

    def fit(self, training):
        x = self._logged(training)
        self.training_rows = len(x)
        self.scaler = StandardScaler().fit(x)
        self.pca = None
        if self.retained_variance is not None:
            z = self.scaler.transform(x)
            if not np.any(self.scaler.var_ > 0):
                raise ValueError("PCA needs at least one nonconstant training feature")
            self.pca = PCA(n_components=self.retained_variance, svd_solver="full",
                           whiten=False).fit(z)
        return self

    def transform(self, values):
        z = self.scaler.transform(self._logged(values))
        if self.pca is not None:
            z = self.pca.transform(z)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite transformed features")
        return z

    def summary(self):
        return {"training_rows": self.training_rows,
                "input_dimensions": len(self.columns),
                "output_dimensions": int(self.pca.n_components_) if self.pca is not None else len(self.columns),
                "retained_variance_requested": self.retained_variance,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist() if self.pca is not None else None,
                "constant_columns": [c for c, v in zip(self.columns, self.scaler.var_) if v == 0],
                "continuous_transform": "sign(x)*log1p(abs(x)); binary columns unchanged before scaling",
                "scaling": "training-only StandardScaler; no clipping", "row_weights": "uniform"}


@dataclass
class Bundle:
    recipe: str
    preprocessor: Preprocessor
    target_order: tuple
    classifiers: tuple

    def predict(self, values, *, columns, target_order=TARGET_ORDER):
        if tuple(columns) != self.preprocessor.columns or tuple(target_order) != self.target_order:
            raise ValueError("Feature or target order differs from the fitted bundle")
        z = self.preprocessor.transform(values)
        predictions = []
        for model in self.classifiers:
            if isinstance(model, int):
                predictions.append(np.full(len(z), float(model)))
            else:
                positive = list(model.classes_).index(1)
                predictions.append(model.predict_proba(z)[:, positive])
        p = np.column_stack(predictions)
        if p.shape != (len(z), len(self.target_order)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid six-target probabilities")
        return p


def fit_recipe(recipe, x, y, columns):
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    x = checked_matrix(x, len(columns))
    y = np.asarray(y)
    if y.shape != (len(x), len(TARGET_ORDER)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected six aligned binary training targets")
    started = time.perf_counter()
    prep = Preprocessor(columns, RECIPES[recipe]).fit(x)
    z = prep.transform(x)
    prep_seconds = time.perf_counter() - started
    classifiers, fits = [], []
    for j, target in enumerate(TARGET_ORDER):
        target_started = time.perf_counter()
        unique = np.unique(y[:, j])
        retry = False
        if len(unique) == 1:
            model = int(unique[0])
            iterations = 0
        else:
            model = (LinearDiscriminantAnalysis(**LDA_PARAMS) if recipe == "fisher_lda"
                     else LogisticRegression(**LR_PARAMS))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(z, y[:, j])
            except ConvergenceWarning:
                if recipe == "fisher_lda":
                    raise
                retry = True
                model.set_params(max_iter=4000)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(z, y[:, j])
            iterations = int(np.max(getattr(model, "n_iter_", [0])))
        classifiers.append(model)
        fits.append({"target": target, "training_rows": len(y),
                     "positive_count": int(y[:, j].sum()), "constant_class": model if isinstance(model, int) else None,
                     "iterations": iterations, "convergence_retry_4000": retry,
                     "fit_seconds": time.perf_counter() - target_started})
    return Bundle(recipe, prep, tuple(TARGET_ORDER), tuple(classifiers)), {
        "recipe": recipe, "preprocessing": prep.summary(), "target_fits": fits,
        "preprocessing_seconds": prep_seconds, "total_fit_seconds": time.perf_counter() - started}
