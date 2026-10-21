"""Run one sort trial in a fresh process and write its measurements as JSON.

Polars reads its ``POLARS_OOC_*`` settings once, when the engine first starts,
so changing ``os.environ`` after ``import polars`` has no effect. Each trial
therefore runs in its own interpreter, launched by ``benchmark.py`` with the
trial's environment already set. Nothing here touches the environment.

Usage::

    python benchmarks/sort_trial.py --method polars --input in.parquet \
        --output out.parquet --by l_partkey --result result.json \
        [--options '{"maintain_order": true}'] [--work-directory DIR] [--no-verify]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["polars", "custom"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--by", action="append", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--options", default="{}")
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    options = json.loads(args.options)

    import polars as pl

    result: dict = {"polars_version": pl.__version__}

    start = time.perf_counter()
    if args.method == "polars":
        pl.scan_parquet(args.input).sort(args.by, **options).sink_parquet(args.output)
    else:
        from utils.external_sort import sort_parquet

        report = sort_parquet(
            args.input,
            args.output,
            args.by,
            work_directory=args.work_directory,
            **options,
        )
        result["buckets"] = report.buckets
        result["largest_bucket_rows"] = report.largest_bucket_rows
    result["sort_seconds"] = time.perf_counter() - start

    # Read before verifying, so the check's own memory is not counted.
    result.update(_peak_memory_mb())

    if not args.no_verify:
        start = time.perf_counter()
        result.update(_verify(pl, args.input, args.output, args.by[0], options))
        result["verify_seconds"] = time.perf_counter() - start

    args.result.write_text(json.dumps(result))


def _verify(pl, source: Path, output: Path, first_key: str, options: dict) -> dict:
    """Row counts match and the leading key never steps backwards."""
    descending = options.get("descending", False)
    if not isinstance(descending, bool):
        descending = descending[0]
    key = pl.col(first_key)
    previous = key.shift(1)
    in_order = (key <= previous) if descending else (key >= previous)
    checked = (
        pl.scan_parquet(output)
        .select(
            pl.len().alias("rows_out"),
            in_order.fill_null(True).all().alias("sorted_ok"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    rows_in = pl.scan_parquet(source).select(pl.len()).collect().item()
    return {"rows_in": rows_in, **checked}


def _peak_memory_mb() -> dict:
    """Peak resident memory and, on Windows, peak committed private memory.

    The working set can be trimmed by the OS under memory pressure, so on
    Windows the peak commit charge is the better measure of what was allocated.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return {}
        return {
            "peak_rss_mb": counters.PeakWorkingSetSize / 2**20,
            "peak_commit_mb": counters.PeakPagefileUsage / 2**20,
        }

    import resource

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    scale = 1 if sys.platform == "darwin" else 1024
    return {"peak_rss_mb": max_rss * scale / 2**20}


if __name__ == "__main__":
    main()
