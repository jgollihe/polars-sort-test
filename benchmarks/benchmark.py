"""Benchmark Polars 2.0.0 OOC sorting against ``utils.external_sort.sort_parquet``.

I originally built this to sort Pseudopeople data but since that requires
permission to utilize, I built this test around the TPCH lineitem dataset.

The first step is to build that dataset, which can be done relatively fast
using the CLI:

```bash
tpchgen-cli parquet -s 60 --tables lineitem --output-dir ./tpch_test --parts 10
```

Then point this script at the parts directory (combined once into
``lineitem_full.parquet`` next to it) or at any single Parquet file:

```bash
python benchmarks/benchmark.py ./tpch_test/lineitem
python benchmarks/benchmark.py data.parquet --by l_partkey --repeats 1 --only "custom_*" \
    --output-dir results/test_run
python benchmarks/benchmark.py data.parquet --methods my_methods.toml --dry-run
```

The methods being compared live in a TOML file (``benchmarks/methods.toml`` by
default); see that file for the format.

Polars reads its ``POLARS_OOC_*`` settings once, when the engine starts, so
setting ``os.environ`` between runs in one process does nothing. Instead every
trial runs ``benchmarks/sort_trial.py`` in a fresh interpreter with that
trial's environment, which also gives each trial its own peak-memory reading
and keeps one trial's allocator state or page cache in Polars from leaking into
the next. Trials are interleaved (every method and column once, then again) so
slow drift in the machine is spread across methods rather than landing on one.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_METHODS_FILE = REPO / "benchmarks" / "methods.toml"
DEFAULT_OUTPUT_DIR = REPO / "results"
RESULTS_FILENAME = "sort_benchmark.csv"
DEFAULT_SORT_COLUMNS = ["l_partkey", "l_shipmode", "l_shipinstruct"]
METHOD_KINDS = ("polars", "custom")

RESULT_FIELDS = [
    "started_at",
    "method",
    "kind",
    "sort_column",
    "repeat",
    "status",
    "sort_seconds",
    "process_seconds",
    "peak_rss_mb",
    "peak_commit_mb",
    "rows_in",
    "rows_out",
    "sorted_ok",
    "buckets",
    "largest_bucket_rows",
    "verify_seconds",
    "returncode",
    "env",
    "options",
    "log",
]


@dataclass(frozen=True)
class Config:
    input_file: Path
    # Polars' spill files, the custom sort's buckets and the sorted output all
    # go here, one subfolder per trial, so both methods read and write the
    # same disk.
    scratch_dir: Path
    results_file: Path
    log_dir: Path
    timeout_seconds: float
    # The check adds a full read of the output to every trial (outside the
    # timing), so turn it off once both methods are trusted.
    verify: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        type=Path,
        help="Parquet file to sort, or a directory of Parquet parts to combine "
        "once into <directory>_full.parquet beside it.",
    )
    parser.add_argument(
        "--by",
        dest="sort_columns",
        action="append",
        metavar="COLUMN",
        help="Column to sort on; repeat for several, each benchmarked separately. "
        f"Default: {', '.join(DEFAULT_SORT_COLUMNS)}.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Default: %(default)s."
    )
    parser.add_argument(
        "--methods",
        dest="methods_file",
        type=Path,
        default=DEFAULT_METHODS_FILE,
        help="TOML file listing the methods to compare. Default: %(default)s.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="PATTERN",
        help="Run only methods whose name matches this glob; repeatable.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="Working space for spill files, buckets and sorted output. "
        "Default: a 'scratch' folder beside the input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for {RESULTS_FILENAME} and per-trial logs; rerunning with "
        "the same folder resumes that run. Default: %(default)s.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=120,
        help="Per-trial time limit. Default: %(default)s.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip checking each trial's output is sorted and complete.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Rerun trials that already have an 'ok' row in the results file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the trials that would run, then exit.",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.input.exists():
        parser.error(f"input {args.input} does not exist")
    return args


def load_methods(path: Path, only: list[str] | None) -> list[dict]:
    """Read the ``[[method]]`` tables from ``path``, keeping those matching ``only``."""
    with path.open("rb") as handle:
        entries = tomllib.load(handle).get("method", [])

    methods = []
    for entry in entries:
        name = entry.get("name")
        kind = entry.get("kind")
        if not name or kind not in METHOD_KINDS:
            raise SystemExit(
                f"{path}: every method needs a name and a kind of "
                f"{' or '.join(METHOD_KINDS)}; got {entry}"
            )
        methods.append(
            {
                "name": name,
                "kind": kind,
                # Environment values must be strings; TOML lets them be numbers.
                "env": {k: str(v) for k, v in entry.get("env", {}).items()},
                "options": entry.get("options", {}),
            }
        )

    names = [m["name"] for m in methods]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(f"{path}: duplicate method names: {', '.join(duplicates)}")

    if only:
        methods = [
            m for m in methods if any(fnmatch.fnmatch(m["name"], p) for p in only)
        ]
    if not methods:
        raise SystemExit(f"No methods to run (file {path}, --only {only}).")
    return methods


def prepare_input(source: Path) -> Path:
    """Return a single Parquet file for ``source``, combining parts once if needed."""
    import polars as pl

    if source.is_dir():
        combined = source.with_name(f"{source.name}_full.parquet")
        if not combined.exists():
            print(f"Combining {source} into {combined}")
            pl.scan_parquet(source).sink_parquet(
                combined, compression="zstd", engine="streaming"
            )
        source = combined
    n_rows = pl.scan_parquet(source).select(pl.len()).collect().item()
    print(f"Input: {source} ({n_rows:,} rows)")
    return source


def trial_env(method: dict, spill_directory: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("POLARS_OOC_")}
    env["POLARS_OOC_SPILL_DIR"] = str(spill_directory)
    # Prints per-node spill counts to the trial log, which shows whether a
    # run actually went out of core.
    env["POLARS_OOC_LOG_METRICS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    env.update(method["env"])
    return env


def run_trial(config: Config, method: dict, sort_column: str, repeat: int) -> dict:
    label = f"{method['name']}-{sort_column}-r{repeat}"
    trial_dir = config.scratch_dir / label
    shutil.rmtree(trial_dir, ignore_errors=True)
    spill_directory = trial_dir / "spill"
    work_directory = trial_dir / "work"
    spill_directory.mkdir(parents=True)
    work_directory.mkdir()
    result_path = trial_dir / "result.json"
    log_path = config.log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.log"

    command = [
        sys.executable,
        str(REPO / "benchmarks" / "sort_trial.py"),
        "--method", method["kind"],
        "--input", str(config.input_file),
        "--output", str(trial_dir / "sorted.parquet"),
        "--by", sort_column,
        "--result", str(result_path),
        "--options", json.dumps(method["options"]),
        "--work-directory", str(work_directory),
    ]  # fmt: skip
    if not config.verify:
        command.append("--no-verify")

    row = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "method": method["name"],
        "kind": method["kind"],
        "sort_column": sort_column,
        "repeat": repeat,
        "env": json.dumps(method["env"]),
        "options": json.dumps(method["options"]),
        "log": str(log_path),
    }
    start = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                env=trial_env(method, spill_directory),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
            )
        row["returncode"] = completed.returncode
        row["status"] = "ok" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    row["process_seconds"] = time.perf_counter() - start

    if row["status"] == "ok":
        row.update(json.loads(result_path.read_text()))
        if config.verify and not (
            row["sorted_ok"] and row["rows_in"] == row["rows_out"]
        ):
            row["status"] = "wrong_output"
    shutil.rmtree(trial_dir, ignore_errors=True)
    return row


def append_result(results_file: Path, row: dict) -> None:
    new_file = not results_file.exists()
    with results_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, RESULT_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def completed_trials(results_file: Path) -> set[tuple[str, str, int]]:
    if not results_file.exists():
        return set()
    with results_file.open(newline="") as handle:
        return {
            (row["method"], row["sort_column"], int(row["repeat"]))
            for row in csv.DictReader(handle)
            if row["status"] == "ok"
        }


def check_results_header(results_file: Path) -> None:
    if not results_file.exists():
        return
    with results_file.open(newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != RESULT_FIELDS:
        raise SystemExit(
            f"{results_file} has different columns from this version of the "
            "benchmark; pass --output-dir with a new folder."
        )


def summarize(results_file: Path) -> None:
    import polars as pl

    results = pl.read_csv(results_file)
    summary = (
        results.group_by("sort_column", "method")
        .agg(
            pl.len().alias("trials"),
            (pl.col("status") == "ok").sum().alias("ok"),
            pl.col("sort_seconds").filter(pl.col("status") == "ok").median().alias("median_s"),
            pl.col("sort_seconds").filter(pl.col("status") == "ok").min().alias("min_s"),
            pl.col("sort_seconds").filter(pl.col("status") == "ok").max().alias("max_s"),
            pl.col("peak_commit_mb").filter(pl.col("status") == "ok").max().alias("peak_commit_mb"),
        )
        .sort("sort_column", "median_s", nulls_last=True)
    )  # fmt: skip
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN"
    ):
        print(summary)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    methods = load_methods(args.methods_file, args.only)
    sort_columns = args.sort_columns or DEFAULT_SORT_COLUMNS
    results_file = args.output_dir / RESULTS_FILENAME
    check_results_header(results_file)

    trials = [
        (method, column, repeat)
        for repeat in range(args.repeats)
        for column in sort_columns
        for method in methods
    ]
    done = completed_trials(results_file) if args.resume else set()

    if args.dry_run:
        for number, (method, column, repeat) in enumerate(trials, start=1):
            skip = " (already done)" if (method["name"], column, repeat) in done else ""
            print(
                f"[{number}/{len(trials)}] {method['name']} on {column}, repeat {repeat}{skip}"
            )
        return

    input_file = prepare_input(args.input.resolve())
    config = Config(
        input_file=input_file,
        scratch_dir=(args.scratch_dir or input_file.parent / "scratch").resolve(),
        results_file=results_file,
        log_dir=args.output_dir / "logs",
        timeout_seconds=args.timeout_minutes * 60,
        verify=args.verify,
    )
    config.scratch_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    for number, (method, column, repeat) in enumerate(trials, start=1):
        if (method["name"], column, repeat) in done:
            print(
                f"[{number}/{len(trials)}] {method['name']} on {column}, repeat {repeat}: already done"
            )
            continue
        print(f"[{number}/{len(trials)}] {method['name']} on {column}, repeat {repeat}")
        row = run_trial(config, method, column, repeat)
        append_result(config.results_file, row)
        seconds = row.get("sort_seconds")
        timing = f"{seconds:.1f}s" if seconds is not None else "-"
        print(f"    {row['status']}: {timing}, log {row['log']}")

    summarize(config.results_file)


if __name__ == "__main__":
    main()
