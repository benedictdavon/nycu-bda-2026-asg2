"""Self-contained synthetic data, with Parquet I/O shim only for offline CI.

The execution environment for these tests may lack PyArrow; production reads/writes
still use actual Parquet. The shim exercises all application semantics, not PyArrow.
"""
from pathlib import Path
import pandas as pd
import pytest


def dt(s):
    return pd.Timestamp(s).tz_convert("UTC").tz_localize(None)


def _fake_evaluator():
    # Minimal contract mirror for isolated tests. Production loads official evaluate.py.
    return '''import pandas as pd
EVENT_COLS=["open_pr","open_issue","issue_comment","pr_review","pr_review_comment","push"]
OUTPUT_COLUMNS=["UserID",*EVENT_COLS]
def read_predictions(path):
    x=pd.read_csv(path)
    assert list(x.columns)==OUTPUT_COLUMNS
    assert x.UserID.is_unique
    assert x[EVENT_COLS].isin([0,1]).all().all()
    return x
def f1_per_event(pred,truth):
    p=pred.set_index("UserID").loc[truth.UserID,EVENT_COLS].to_numpy()
    t=truth[EVENT_COLS].to_numpy()
    tp=((p==1)&(t==1)).sum(0); fp=((p==1)&(t==0)).sum(0); fn=((p==0)&(t==1)).sum(0)
    return [float(2*a/(2*a+b+c) if 2*a+b+c else 1) for a,b,c in zip(tp,fp,fn)]
def scores(pred,truth):
    p=pred.set_index("UserID").loc[truth.UserID,EVENT_COLS].to_numpy()
    t=truth[EVENT_COLS].to_numpy()
    tp=((p==1)&(t==1)).sum(0); fp=((p==1)&(t==0)).sum(0); fn=((p==0)&(t==1)).sum(0)
    macro=sum(f1_per_event(pred,truth))/6
    a,b,c=tp.sum(),fp.sum(),fn.sum()
    micro=float(2*a/(2*a+b+c) if 2*a+b+c else 1)
    return {"macro_f1":float(macro),"micro_f1":micro,"score":float((macro+micro)/2)}
'''


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    from src.data_loader import OfficialData

    root = tmp_path / "personal"
    db = root / "assignment2/data/db"
    db.mkdir(parents=True)
    (root / "assignment2/evaluation/examples").mkdir(parents=True)
    (root / "assignment2/evaluation/evaluate.py").write_text(_fake_evaluator())
    r = 42
    before = dt("2026-09-25T10:00:00Z")
    tables = {
        "Users": pd.DataFrame({
            "user_id": [0, 1, 2, 3],
            "github_user_id": [101, 202, 303, 999],
            "is_cohort": [True, True, True, False],
            "first_seen_at": [dt("2025-12-01T00:00Z")] * 4,
        }),
        "Repositories": pd.DataFrame({
            "repo_id": [r], "first_seen_at": [before],
        }),
        "PullRequests": pd.DataFrame({
            "author_id": [0, 1], "repo_id": [r, r],
            "created_at": [dt("2026-09-27T13:00:00Z"), dt("2026-10-04T13:00:00Z")],
            "first_seen_at": [dt("2026-10-07T08:00:00Z"), dt("2026-10-04T13:00:00Z")],
        }),
        "Issues": pd.DataFrame({
            "author_id": [0, 1, 1], "repo_id": [r, r, r],
            "is_pull_request": [False, False, True],
            "created_at": [dt("2026-09-28T03:08:24Z"), dt("2026-09-28T18:38:47Z"), dt("2026-09-30T00:00:00Z")],
            "first_seen_at": [dt("2026-10-09T05:46:50Z"), dt("2026-10-10T05:11:20Z"), dt("2026-09-30T00:00:00Z")],
        }),
        "IssueComments": pd.DataFrame({
            "author_id": [1], "repo_id": [r], "created_at": [dt("2026-09-28T00:00:00Z")],
        }),
        "PRReviews": pd.DataFrame({
            "reviewer_id": [1], "repo_id": [r],
            "first_seen_at": [dt("2026-09-28T06:00:00Z")],
            "submitted_at": [dt("2026-09-28T06:00:00Z")],
        }),
        "PRReviewComments": pd.DataFrame({
            "author_id": [0], "repo_id": [r], "created_at": [dt("2026-09-29T00:00:00Z")],
        }),
        "Pushes": pd.DataFrame({
            "user_id": [0, 1, 0, 3], "repo_id": [r]*4,
            "created_at": [dt("2026-09-27T13:00:00Z"), dt("2026-09-27T12:59:59Z"),
                           dt("2026-10-04T13:00:00Z"), dt("2026-09-29T00:00:00Z")],
        }),
    }
    for name, df in tables.items():
        # Only for tests; do not change actual Parquet functionality in delivered code.
        df.to_pickle(db / f"{name}.parquet")
    original_read = pd.read_parquet
    original_write = pd.DataFrame.to_parquet

    def fake_read(path, columns=None, **kwargs):
        df = pd.read_pickle(path)
        return df if columns is None else df[columns]

    def fake_write(df, path, *args, **kwargs):
        df.to_pickle(path)

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_write)
    src = OfficialData(db, expected_cohort=3)
    return {"root": root, "db": db, "source": src, "tables": tables}
