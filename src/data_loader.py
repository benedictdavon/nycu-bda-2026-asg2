"""Read only the instructor-provided Parquet snapshot and map stable GitHub user IDs.

No web/API collection occurs. Numeric PKs and foreign keys are used only within one
snapshot; github_user_id is the sole stable public user key.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
from typing import Optional

import pandas as pd

TABLES = (
    "Users", "Repositories", "PullRequests", "Issues", "IssueComments",
    "PRReviews", "PRReviewComments", "Pushes",
)
TARGETS = (
    "open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push",
)
ROW_TIME = {
    "Users": "first_seen_at", "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at", "Issues": "first_seen_at",
    "IssueComments": "created_at", "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at", "Pushes": "created_at",
}


@dataclass(frozen=True)
class TargetSpec:
    table: str
    actor_fk: str
    observed_column: str
    event_column: str
    non_pr_only: bool = False


SPECS = {
    "open_pr": TargetSpec("PullRequests", "author_id", "first_seen_at", "created_at"),
    "open_issue": TargetSpec("Issues", "author_id", "first_seen_at", "created_at", True),
    "issue_comment": TargetSpec("IssueComments", "author_id", "created_at", "created_at"),
    "pr_review": TargetSpec("PRReviews", "reviewer_id", "first_seen_at", "submitted_at"),
    "pr_review_comment": TargetSpec("PRReviewComments", "author_id", "created_at", "created_at"),
    "push": TargetSpec("Pushes", "user_id", "created_at", "created_at"),
}
LATE_KNOWN = frozenset({"merged_at", "merged_by_id", "closed_at", "state_reason", "dismissed_at"})


def utc(value) -> pd.Timestamp:
    """Interpret all timezone-naive input timestamps as UTC, never as local time."""
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Timestamp cannot be NaT")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def locate_data_dir(project_root: Optional[str | Path] = None,
                    data_dir: Optional[str | Path] = None) -> Path:
    """Find the eight official files without modifying the course repository."""
    bases = []
    if data_dir:
        bases.extend([Path(data_dir), Path(data_dir) / "db", Path(data_dir) / "data/db"])
    if project_root:
        p = Path(project_root)
        bases.extend([p / "assignment2/data/db", p / "data/db"])
    if not bases:
        for p in (Path.cwd(), *Path.cwd().parents):
            bases.append(p / "assignment2/data/db")
    for base in bases:
        base = base.expanduser().resolve()
        if all((base / f"{t}.parquet").is_file() for t in TABLES):
            return base
    raise FileNotFoundError(
        "Could not find all eight instructor Parquet files. Clone the official repository "
        "into assignment2/, or specify --data-dir PATH (the directory with *.parquet)."
    )


def repo_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_official_evaluator(repo: Path):
    path = repo / "evaluation" / "evaluate.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official evaluator not found at {path}")
    spec = importlib.util.spec_from_file_location("asg2_official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if module.EVENT_COLS != list(TARGETS) or module.OUTPUT_COLUMNS != ["UserID", *TARGETS]:
        raise ValueError("Official evaluator targets changed; inspect before continuing")
    return module


class OfficialData:
    """Lazy per-table reader. Target-event frames contain only cohort-associated actors."""

    def __init__(self, data_dir: str | Path, *, expected_cohort: int | None = 9000):
        self.data_dir = Path(data_dir).resolve()
        self.official_repo = self.data_dir.parent.parent
        self.snapshot_sha = repo_commit(self.official_repo)
        self._events: dict[str, pd.DataFrame] = {}
        self._max_row_seen: pd.Timestamp | None = None
        self.users = self._read("Users", [
            "user_id", "github_user_id", "is_cohort", "first_seen_at",
        ])
        if self.users.user_id.isna().any() or not self.users.user_id.is_unique:
            raise ValueError("Users.user_id must be a unique, non-null within-snapshot PK")
        if self.users.github_user_id.isna().any() or not self.users.github_user_id.is_unique:
            raise ValueError("Users.github_user_id must be unique and non-null")
        self.cohort = self.users.loc[self.users.is_cohort.eq(True)].copy()
        if expected_cohort is not None and len(self.cohort) != expected_cohort:
            raise ValueError(f"Expected {expected_cohort} cohort users, got {len(self.cohort)}")
        self.cohort["UserID"] = self.cohort.github_user_id.astype("int64")
        self.cohort["first_seen_utc"] = pd.to_datetime(
            self.cohort.first_seen_at, utc=True, errors="coerce"
        )
        self.cohort = self.cohort.sort_values("UserID").reset_index(drop=True)
        self.cohort_ids = pd.Index(self.cohort.UserID, name="UserID")
        self.actor_to_public = self.cohort.set_index("user_id")["UserID"]

    def _read(self, table: str, columns: list[str]) -> pd.DataFrame:
        if any(c in LATE_KNOWN for c in columns):
            raise ValueError("Late-known columns are intentionally unavailable to this pipeline")
        path = self.data_dir / f"{table}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            return pd.read_parquet(path, columns=columns)
        except ImportError as exc:
            raise RuntimeError("Parquet support missing. Activate environment.yaml (PyArrow).") from exc

    def events(self, target: str) -> pd.DataFrame:
        if target not in SPECS:
            raise KeyError(f"Unknown activity target {target!r}")
        if target in self._events:
            return self._events[target]
        s = SPECS[target]
        cols = list(dict.fromkeys([s.actor_fk, s.observed_column, s.event_column, "repo_id"]
                                  + (["is_pull_request"] if s.non_pr_only else [])))
        d = self._read(s.table, cols)
        if s.non_pr_only:
            d = d.loc[d.is_pull_request.eq(False)]
        d = d.loc[d[s.actor_fk].isin(self.actor_to_public.index)].copy()
        d["UserID"] = d[s.actor_fk].map(self.actor_to_public).astype("int64")
        d["seen_utc"] = pd.to_datetime(d[s.observed_column], utc=True, errors="coerce")
        d["event_utc"] = pd.to_datetime(d[s.event_column], utc=True, errors="coerce")
        if d.seen_utc.isna().any():
            raise ValueError(f"{s.table}: missing official row-visibility time")
        d = d[["UserID", "repo_id", "seen_utc", "event_utc"]].reset_index(drop=True)
        self._events[target] = d
        return d

    def latest_observed(self) -> pd.Timestamp:
        """Observed row-time maximum, NOT a certified collection-completeness watermark."""
        if self._max_row_seen is None:
            # All six event tables plus Users. Repositories rarely pushes a newer maximum.
            values = [self.cohort.first_seen_utc.max()]
            for name in TARGETS:
                values.append(self.events(name).seen_utc.max())
            self._max_row_seen = max(v for v in values if pd.notna(v))
        return self._max_row_seen

    def fingerprint(self) -> dict:
        files = {}
        for t in TABLES:
            p = self.data_dir / f"{t}.parquet"
            stat = p.stat()
            files[t] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {
            "official_sha": self.snapshot_sha, "data_directory": str(self.data_dir),
            "cohort_users": len(self.cohort), "latest_observed_utc": str(self.latest_observed()),
            "files": files,
        }

    def official_test_user_ids(self, path: str | Path | None = None) -> pd.Index:
        path = Path(path) if path is not None else self.official_repo / "evaluation/test_input.csv"
        if not path.is_file():
            raise FileNotFoundError(f"User ID list not found (official test users arrive Oct 28): {path}")
        frame = pd.read_csv(path)
        if list(frame.columns) != ["UserID"]:
            raise ValueError("Test user CSV must have exactly one column, UserID")
        if frame.UserID.isna().any() or frame.UserID.duplicated().any():
            raise ValueError("Test UserID values must be unique and non-null")
        ids = pd.Index(frame.UserID.astype("int64"), name="UserID")
        unknown = ids.difference(self.cohort_ids)
        if len(unknown):
            raise ValueError(f"{len(unknown)} test users are not in instructor cohort")
        return ids
