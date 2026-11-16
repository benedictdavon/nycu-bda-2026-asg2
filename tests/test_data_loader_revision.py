import pandas as pd
import pytest

from src.data_loader import load_official_evaluator


def test_revised_roster_default_precedes_legacy_roster(synthetic):
    source = synthetic["source"]
    revised = source.official_repo / "submission" / "test_input.csv"
    revised.parent.mkdir()
    pd.DataFrame({"UserID": [202, 101]}).to_csv(revised, index=False)
    pd.DataFrame({"UserID": [303]}).to_csv(source.official_repo / "evaluation/test_input.csv", index=False)
    assert source.official_test_user_ids().tolist() == [202, 101]


def test_explicit_roster_and_unknown_revised_users(synthetic):
    source = synthetic["source"]
    roster = source.official_repo / "custom.csv"
    pd.DataFrame({"UserID": [303]}).to_csv(roster, index=False)
    assert source.official_test_user_ids(roster).tolist() == [303]
    pd.DataFrame({"UserID": [123456]}).to_csv(roster, index=False)
    with pytest.raises(ValueError, match="missing from the loaded cohort"):
        source.official_test_user_ids(roster)


def test_revised_evaluator_default_precedes_legacy(synthetic):
    course = synthetic["source"].official_repo
    revised = course / "scripts" / "evaluate.py"
    revised.parent.mkdir()
    revised.write_text((course / "evaluation/evaluate.py").read_text(), encoding="utf-8")
    assert load_official_evaluator(course).__file__ == str(revised)
