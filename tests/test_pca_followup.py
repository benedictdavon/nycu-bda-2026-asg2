"""Fixed EW7 mass, authenticated representation reuse and classifier-weight contracts."""
import numpy as np
import pandas as pd
import pytest

from experiments.neural_research_common.data import TARGET_ORDER
from experiments.pca_fisher_20261026.models import fit_recipe
from experiments.pca_followup_20261026.models import (
    WEIGHTED_RECIPES, fixed_rank_scores, fit_weighted_lr,
)
from experiments.pca_followup_20261026.verify import construct_candidates
from src.phase_b_rules import blend_score, rank01


def test_ensemble_endpoints_reproduce_existing_p021():
    rng = np.random.default_rng(72)
    h, p, ew = [rng.random((25, 6)) for _ in range(3)]
    for name, weights in (("h", {"h": 1.0}), ("p", {"p": 1.0})):
        model_score, final = fixed_rank_scores({"h": h, "p": p}, ew, weights)
        expected = h if name == "h" else p
        np.testing.assert_array_equal(final, blend_score(expected, ew, 0.5))
        np.testing.assert_array_equal(model_score, rank01(expected))


def test_intermediate_rank_mixture_is_not_ranked_again():
    h = np.tile(np.array([0.1, 0.2, 0.3, 0.4])[:, None], (1, 6))
    p = np.tile(np.array([0.2, 0.1, 0.3, 0.4])[:, None], (1, 6))
    ew = h[::-1]
    mixed, final = fixed_rank_scores({"h": h, "p": p}, ew, {"h": 0.75, "p": 0.25})
    np.testing.assert_allclose(final - 0.5 * mixed, 0.5 * rank01(ew), rtol=0, atol=1e-15)
    assert not np.array_equal(final, blend_score(mixed, ew, 0.5))


@pytest.mark.parametrize("weights", ({"h": 0.6}, {"h": 1.2, "p": -0.2}, {"h": np.nan}))
def test_invalid_mixture_weights_are_rejected(weights):
    x = np.full((4, 6), 0.5)
    with pytest.raises(ValueError, match="weights"):
        fixed_rank_scores({"h": x, "p": x}, x, weights)


@pytest.fixture
def fitted_pca():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(240, 5))
    y = (rng.random((240, 6)) < 0.4).astype(np.int8)
    y[:, 0], y[:, 1] = 0, 1
    columns = [f"feature_{i}" for i in range(5)]
    bundle, _ = fit_recipe("pca99_lr", x, y, columns)
    return x, y, columns, bundle


def test_uniform_weight_limit_matches_parent_and_preserves_preprocessing(fitted_pca):
    x, y, columns, original = fitted_pca
    means = original.preprocessor.scaler.mean_.copy()
    components = original.preprocessor.pca.components_.copy()
    fitted, summary = fit_weighted_lr("uniform_limit", 1.0, original.preprocessor, x, y, np.ones(len(x)), columns)
    np.testing.assert_allclose(fitted.predict(x, columns=columns), original.predict(x, columns=columns), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fitted.preprocessor.scaler.mean_, means)
    np.testing.assert_array_equal(fitted.preprocessor.pca.components_, components)
    assert summary["preprocessing_reused"] is True
    np.testing.assert_array_equal(fitted.predict(x[:5], columns=columns)[:, :2], np.tile([0, 1], (5, 1)))


def test_nonuniform_weights_change_classifier_without_changing_pca(fitted_pca):
    x, y, columns, original = fitted_pca
    weights = np.linspace(0.05, 3.0, len(x))
    weights /= weights.mean()
    fitted, _ = fit_weighted_lr("weighted", 1.0, original.preprocessor, x, y, weights, columns)
    assert np.max(np.abs(fitted.predict(x, columns=columns) - original.predict(x, columns=columns))) > 0.01
    np.testing.assert_array_equal(fitted.preprocessor.pca.components_, original.preprocessor.pca.components_)


@pytest.mark.parametrize("bad", (np.ones(5), np.zeros(240), np.full(240, np.nan), np.full(240, 2.0)))
def test_invalid_sample_weights_are_rejected(fitted_pca, bad):
    x, y, columns, original = fitted_pca
    with pytest.raises(ValueError, match="Weights"):
        fit_weighted_lr("bad", 1.0, original.preprocessor, x, y, bad, columns)


def test_constructed_candidates_preserve_counts_and_align_constituents():
    rng = np.random.default_rng(75)
    n = 14
    controls = pd.DataFrame({"origin": ["week-a"] * 7 + ["week-b"] * 7,
                             "UserID": list(range(7)) * 2})
    p = rng.random((n, 6))
    for j, t in enumerate(TARGET_ORDER):
        controls[f"p_H2_{t}"] = p[:, j]
        controls[f"ew7_{t}"] = rng.random(n)
        controls[f"persist_{t}"] = np.arange(n) % 3 == 0
        controls[f"y_{t}"] = rng.integers(0, 2, n)
    parent, weighted = [], []
    for name in ("lr", "pca99_lr", "fisher_lda", *WEIGHTED_RECIPES):
        block = controls[["origin", "UserID"]].copy()
        block["recipe"] = name
        for t in TARGET_ORDER:
            block[f"p_{t}"] = rng.random(n)
        (weighted if name in WEIGHTED_RECIPES else parent).append(block.iloc[::-1])
    candidates = construct_candidates(controls, pd.concat(parent), pd.concat(weighted))
    assert not candidates.duplicated(["recipe", "origin", "UserID"]).any()
    for (recipe, origin), group in candidates.groupby(["recipe", "origin"]):
        source = controls.loc[controls.origin == origin]
        for t in TARGET_ORDER:
            assert group[f"dec_{t}"].sum() == source[f"persist_{t}"].sum()
            assert np.isfinite(group[f"decision_score_{t}"]).all()


def test_duplicate_constituent_keys_fail_before_scoring():
    from experiments.pca_followup_20261026.verify import align
    controls = pd.DataFrame({"origin": ["a", "a"], "UserID": [1, 2]})
    duplicate = pd.DataFrame({"origin": ["a", "a"], "UserID": [1, 1]})
    with pytest.raises(AssertionError, match="keys"):
        align(duplicate, controls)
