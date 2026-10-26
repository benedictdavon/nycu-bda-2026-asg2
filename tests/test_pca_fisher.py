"""Leakage, degeneracy, replay and uncertainty contracts for the PCA/Fisher run."""
import inspect

import joblib
import numpy as np
import pytest

from experiments.neural_research_common.data import BINARY_COLUMNS, TARGET_ORDER
from experiments.pca_fisher_20261026.models import Preprocessor, RECIPES, fit_recipe
from experiments.pca_fisher_20261026.verify import cluster_bootstrap, combined_from_contributions
from src.metrics import confusion_arrays, scores_from_counts


@pytest.fixture
def inputs():
    rng = np.random.default_rng(217)
    x = rng.normal(size=(320, 7))
    x[:, 4] = x[:, 0]  # exact collinearity
    x[:, 5] = 4.0     # constant feature
    x[:, 6] = rng.integers(0, 2, len(x))
    columns = ["count", "delta", "recency", "other", "duplicate", "constant", BINARY_COLUMNS[0]]
    y = np.column_stack([x[:, 0] + rng.normal(size=len(x)) > j / 8 for j in range(6)]).astype(np.int8)
    y[:, 0] = 0
    y[:, 1] = 1
    return x, y, columns


def test_preprocessing_fits_training_only_and_preserves_sentinel(inputs):
    x, _, columns = inputs
    prep = Preprocessor(columns, 0.95).fit(x)
    mean = prep.scaler.mean_.copy()
    components = prep.pca.components_.copy()
    held_out = x[:3].copy()
    held_out[:, :5] = 1e12
    prep.transform(held_out)
    np.testing.assert_array_equal(prep.scaler.mean_, mean)
    np.testing.assert_array_equal(prep.pca.components_, components)
    expected = np.sign(x[:, 0]) * np.log1p(np.abs(x[:, 0]))
    np.testing.assert_allclose(prep.scaler.mean_[0], expected.mean())
    assert prep.scaler.mean_[-1] == x[:, -1].mean()
    assert prep.scaler.scale_[5] == 1
    sentinel = x[:1].copy()
    sentinel[0, 2] = -1
    assert prep._logged(sentinel)[0, 2] == -np.log(2)
    assert list(inspect.signature(Preprocessor.fit).parameters) == ["self", "training"]


@pytest.mark.parametrize("recipe", RECIPES)
def test_six_target_fit_degeneracy_and_exact_joblib_replay(recipe, inputs, tmp_path):
    x, y, columns = inputs
    bundle, info = fit_recipe(recipe, x, y, columns)
    p = bundle.predict(x[:20], columns=columns)
    assert p.shape == (20, 6)
    assert np.all(p[:, 0] == 0) and np.all(p[:, 1] == 1)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert [r["target"] for r in info["target_fits"]] == list(TARGET_ORDER)
    destination = tmp_path / "model.joblib"
    joblib.dump(bundle, destination)
    np.testing.assert_array_equal(joblib.load(destination).predict(x[:20], columns=columns), p)
    other, _ = fit_recipe(recipe, x, y, columns)
    np.testing.assert_array_equal(other.predict(x[:20], columns=columns), p)


def test_label_changes_cannot_change_scaler_or_pca(inputs):
    x, y, columns = inputs
    original, _ = fit_recipe("pca95_lr", x, y, columns)
    shuffled, _ = fit_recipe("pca95_lr", x, y[::-1], columns)
    np.testing.assert_array_equal(original.preprocessor.scaler.mean_, shuffled.preprocessor.scaler.mean_)
    np.testing.assert_array_equal(original.preprocessor.pca.components_, shuffled.preprocessor.pca.components_)


def test_rejects_reordered_columns_and_targets(inputs):
    x, y, columns = inputs
    bundle, _ = fit_recipe("lr", x, y, columns)
    with pytest.raises(ValueError, match="order"):
        bundle.predict(x, columns=columns[::-1])
    with pytest.raises(ValueError, match="order"):
        bundle.predict(x, columns=columns, target_order=TARGET_ORDER[::-1])


def test_invalid_data_cannot_enter_preprocessing_or_labels(inputs):
    x, y, columns = inputs
    bad = x.copy()
    bad[0, -1] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        Preprocessor(columns).fit(bad)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Preprocessor(columns).fit(bad)
    with pytest.raises(ValueError, match="target"):
        Preprocessor(["count", "open_pr"])
    with pytest.raises(ValueError, match="six aligned"):
        fit_recipe("lr", x, y[:, :5], columns)


def test_all_constant_pca_fails_explicitly():
    with pytest.raises(ValueError, match="nonconstant"):
        Preprocessor(["constant"], 0.95).fit(np.ones((20, 1)))


def test_fisher_handles_all_constant_features_with_both_classes():
    x = np.ones((40, 2))
    y = np.tile(np.arange(40)[:, None] % 2, (1, 6))
    bundle, _ = fit_recipe("fisher_lda", x, y, ["a", "b"])
    np.testing.assert_allclose(bundle.predict(x, columns=["a", "b"]), 0.5)


def test_bootstrap_keeps_all_week_rows_for_each_sampled_user():
    rng = np.random.default_rng(34)
    ids = np.array([11, 22, 33, 44, 11, 22, 33])
    y = rng.integers(0, 2, (len(ids), 6))
    candidate = rng.integers(0, 2, y.shape)
    baseline = rng.integers(0, 2, y.shape)
    draws, summary = cluster_bootstrap(ids, y, candidate, baseline, draws=12, seed=88)
    users = np.unique(ids)
    sampler = np.random.default_rng(88)
    for i in range(12):
        selected = sampler.integers(0, len(users), size=len(users))
        indices = np.concatenate([np.flatnonzero(ids == users[j]) for j in selected])
        expected = [scores_from_counts(confusion_arrays(d[indices], y[indices]))["score"]
                    for d in (candidate, baseline)]
        assert draws.iloc[i].delta == pytest.approx(expected[0] - expected[1], abs=1e-14)
    assert summary["clusters"] == 4


def test_bootstrap_metric_matches_official_zero_division_convention():
    counts = np.zeros((2, 3, 6), dtype=np.int64)
    np.testing.assert_array_equal(combined_from_contributions(counts), [1.0, 1.0])
