"""Build official tables, refit twice and package a reproducible prediction CSV."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOUR_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{1,2})\.json\.gz$")
OFFICIAL_ORIGIN = datetime(2026, 11, 21, 16, tzinfo=timezone.utc)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("The origin must include a UTC offset")
    return result.astimezone(timezone.utc)


def inventory(raw: Path) -> list[dict]:
    records = []
    for path in raw.glob("*.json.gz"):
        match = HOUR_NAME.fullmatch(path.name)
        if not match:
            continue
        hour = datetime.fromisoformat(match[1]).replace(hour=int(match[2]), tzinfo=timezone.utc)
        if hour < datetime(2025, 12, 1, tzinfo=timezone.utc):
            raise ValueError(f"The official preprocessor ignores pre-December-2025 input: {path.name}")
        records.append({"name": path.name, "hour_utc": hour.isoformat(),
                        "sha256": digest(path), "bytes": path.stat().st_size})
    if not records:
        raise ValueError(f"No GH Archive hour files in {raw}")
    records.sort(key=lambda row: row["hour_utc"])
    hours = [row["hour_utc"] for row in records]
    if len(hours) != len(set(hours)):
        raise ValueError("Duplicate UTC archive hours (for example, -0 and -00)")
    return records


def run(command: list[str], log: Path) -> None:
    """Stream progress while retaining the exact subprocess output."""
    print("Running:", subprocess.list2cmdline(command), flush=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
        if process.wait():
            raise RuntimeError(f"Command failed; inspect {log}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--origin", type=utc_time, default=OFFICIAL_ORIGIN)
    parser.add_argument("--policy", choices=["h2_p021", "h2wpca_p021", "h2dfs_p021", "persist7"], default="h2_p021")
    parser.add_argument("--rehearsal", action="store_true", help="Label a historical run; never an E3 submission")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{9}", args.student_id):
        parser.error("Use your nine-digit student ID")
    if args.origin != OFFICIAL_ORIGIN and not args.rehearsal:
        parser.error("A historical origin requires --rehearsal")
    course, raw, output = (path.resolve() for path in (args.official_repo, args.raw, args.output_dir))
    required = [course / "scripts" / name for name in ("preprocess.py", "evaluate.py", "cohort.txt")]
    roster = course / "submission" / "test_input.csv"
    if any(not path.is_file() for path in [*required, roster]):
        parser.error("Use the revised course checkout with scripts/ and submission/test_input.csv")
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    records = inventory(raw)
    if not args.rehearsal:
        cutoff = args.origin - timedelta(hours=4)
        if datetime.now(timezone.utc) < cutoff:
            parser.error("Final data cutoff has not occurred. Use --rehearsal for a historical verification.")
        if any(utc_time(row["hour_utc"]) >= cutoff for row in records):
            parser.error("Final raw input must contain only hours before the feature cutoff")
        if utc_time(records[-1]["hour_utc"]) < cutoff - timedelta(hours=1):
            parser.error("Raw input stops before the final pre-cutoff hour; refresh the archive")
    output.mkdir(parents=True)
    manifest = {"status": "running", "rehearsal": args.rehearsal,
                "student_id": args.student_id, "policy": args.policy,
                "origin_utc": args.origin.isoformat(), "raw_directory": str(raw),
                "raw_hours": records, "official_files": {
                    str(path.relative_to(course)): digest(path) for path in [*required, roster]},
                "course_commit": subprocess.check_output(
                    ["git", "-C", str(course), "rev-parse", "HEAD"], text=True).strip(),
                "python": sys.version, "entrypoint_sha256": digest(Path(__file__)),
                "requirements_sha256": digest(ROOT / "requirements-final.txt")}
    receipt = output / "reproduction.json"
    receipt.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        run([sys.executable, str(required[0]), "--raw", str(raw), "--out", str(output / "prepared")],
            output / "preprocess.log")
        predictions = []
        for number in (1, 2):
            prediction = output / f"repeat_{number}" / f"output_{args.student_id}.csv"
            run([sys.executable, "-m", "experiments.final_submission.run", "--mode", "submit",
                 "--origin", args.origin.isoformat(), "--official-repo", str(course),
                 "--data-dir", str(output / "prepared" / "db"), "--roster", str(roster),
                 "--policy", args.policy, "--out", str(prediction),
                 "--output-root", str(output / f"runs_{number}")], output / f"repeat_{number}.log")
            predictions.append(prediction)
        hashes = [digest(path) for path in predictions]
        if hashes[0] != hashes[1]:
            raise RuntimeError("Independent refits produced different CSVs; preserve both for inspection")
        delivered = output / f"output_{args.student_id}.csv"
        delivered.write_bytes(predictions[0].read_bytes())
        manifest.update(status="verified", csv_sha256=hashes[0], independent_refits=2,
                        csv=str(delivered), verified_utc=datetime.now(timezone.utc).isoformat())
        if args.rehearsal:
            (output / "REHEARSAL_ONLY.txt").write_text(
                "Historical reproducibility rehearsal. Do not submit this CSV to E3.\n", encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFIED identical independent refits: {delivered}\nSHA256 {hashes[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
