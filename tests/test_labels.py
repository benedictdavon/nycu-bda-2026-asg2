import pandas as pd
import pytest

from src.data_loader import utc, TARGETS, load_official_evaluator, locate_data_dir
from src.labels import LabelBuilder, verify_sample_truth

START = "2026-09-27T13:00:00Z"
ASOF = "2026-10-09T12:00:00Z"


def test_exact_sample_boundary_and_observation_policy(synthetic):
    src = synthetic["source"]
    result = LabelBuilder(src).build(START, label_as_of=ASOF).set_index("UserID")
    assert result.loc[101, "open_pr"] == 1  # PR created in week, discovered later
    assert result.loc[101, "open_issue"] == 1  # observed by Oct 9
    assert result.loc[202, "open_issue"] == 0  # same week, discovered Oct 10
    assert result.loc[202, "open_pr"] == 0     # event exactly at label end
    assert result.loc[101, "push"] == 1
    assert result.loc[202, "push"] == 0        # event before start
    assert result.loc[202, "issue_comment"] == 1
    assert result.loc[202, "pr_review"] == 1
    assert result.loc[101, "pr_review_comment"] == 1
    assert result.loc[303, list(TARGETS)].sum() == 0


def test_label_cutoff_validation(synthetic):
    lb = LabelBuilder(synthetic["source"])
    with pytest.raises(ValueError, match="label_as_of"):
        lb.build(START, label_as_of="2026-10-01T00:00:00Z")
    with pytest.raises(ValueError, match="Non-cohort"):
        lb.build(START, label_as_of=ASOF, user_ids=[999])
    with pytest.raises(ValueError, match="unique"):
        lb.build(START, label_as_of=ASOF, user_ids=[101,101])


def test_exact_reconstruction_uses_official_evaluator_contract(synthetic):
    src = synthetic["source"]
    truth = LabelBuilder(src).build(START, label_as_of=ASOF)[["UserID", *TARGETS]]
    path = src.official_repo / "evaluation/examples/sample_truth.csv"
    truth.to_csv(path, index=False)
    report = verify_sample_truth(src, start=START, label_as_of=ASOF)
    assert report["total_mismatches"] == 0
    assert report["sample_users"] == 3
    assert report["scores"]["macro_f1"] == 1
    assert report["scores"]["micro_f1"] == 1
    report_wrong = verify_sample_truth(src, start=START, label_as_of="2026-10-11T12:00:00Z")
    assert report_wrong["total_mismatches"] == 1
    assert report_wrong["mismatches"]["open_issue"] == 1


def test_official_test_user_ids(synthetic):
    src = synthetic["source"]
    path = src.official_repo / "evaluation/test_input.csv"
    pd.DataFrame({"UserID": [101, 202]}).to_csv(path, index=False)
    assert list(src.official_test_user_ids()) == [101, 202]
    pd.DataFrame({"UserID": [101, 999]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cohort"):
        src.official_test_user_ids()


def test_data_loader_safety(synthetic):
    src = synthetic["source"]
    assert locate_data_dir(synthetic["root"]) == synthetic["db"]
    assert list(src.cohort_ids) == [101,202,303]
    with pytest.raises(ValueError, match="Late-known"):
        src._read("Issues", ["closed_at"])
    assert utc("2026-11-12T00:00:00+08:00") == pd.Timestamp("2026-11-11T16:00:00Z")
    module = load_official_evaluator(src.official_repo)
    assert module.EVENT_COLS == list(TARGETS)
