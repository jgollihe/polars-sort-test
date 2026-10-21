"""Sort a Parquet file that does not fit in memory.

Polars <2.0.0 cannot spill a sort to disk: ``scan_parquet(...).sort(...).sink_parquet``
gathers the whole sorted set in RAM before writing a row. This module sorts by
range partitioning instead, so that peak memory is one bucket rather than the
whole file:

1. Sample the sort key with each row's original row number, and read evenly
   spaced boundaries off the sample sorted by ``(key, row number)``.
2. Stream the file once, routing each row to the bucket its ``(key, row
   number)`` falls in. Every row in bucket ``k`` sorts before every row in
   bucket ``k + 1``. Because the row number is unique, a heavily repeated key
   is split across as many buckets as it needs, so buckets stay near the
   target size however few distinct keys there are.
3. Sort each bucket in memory by ``(key, row number)``, so the result is the
   stable sort Polars produces with ``maintain_order=True``.
4. Concatenate the buckets in order. No merge step is needed.

Any column type that supports ``<`` works as a key, with the same ``descending``
and ``nulls_last`` semantics as :meth:`polars.LazyFrame.sort`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ROW_INDEX_COLUMN = "__sort_row_index"
BUCKET_COLUMN = "__sort_bucket"


@dataclass(frozen=True)
class SortReport:
    """What :func:`sort_parquet` wrote."""

    rows: int
    buckets: int
    largest_bucket_rows: int
    """Rows in the biggest bucket, which sets the peak memory of the sort."""


def sort_parquet(
    source: Path | str | pl.LazyFrame,
    destination: Path | str,
    by: str | Sequence[str],
    *,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    bucket_rows: int = 20_000_000,
    sample_rows: int = 1_000_000,
    work_directory: Path | str | None = None,
) -> SortReport:
    """Write ``source`` to ``destination`` stably sorted by ``by``.

    The output equals ``pl.read_parquet(source).sort(by, descending=descending,
    nulls_last=nulls_last, maintain_order=True)`` for any file, but only one
    bucket of roughly ``bucket_rows`` rows is ever held in memory. That bound
    holds even when a few keys cover most of the file, since equal keys may be
    split across buckets by their row number.

    Args:
        source: The Parquet file to sort, or a lazy plan. A plan is run up to
            three times (row count, key sample, scatter), so keep it to a scan
            plus cheap projections.
        destination: Where to write the sorted file. Its parent is created.
        by: The sort key column(s), most significant first.
        descending: Sort direction, one flag for all keys or one per key.
        nulls_last: Null placement, one flag for all keys or one per key.
        bucket_rows: Target rows per bucket, which bounds the memory used by
            the in-memory sort of each bucket. A file at or under this size is
            sorted in one step with no scatter.
        sample_rows: Rows sampled from the key columns to choose the bucket
            boundaries. More gives evener buckets; the sample is sorted in
            memory. Keep it well above the bucket count, as each bucket's size
            is estimated from about ``sample_rows / buckets`` sampled rows.
        work_directory: Where the scattered buckets live during the sort. Must
            have room for a second copy of the file. Defaults to the
            destination's directory, so the work lands on the same disk.

    Returns:
        The rows written, the number of buckets used, and the size of the
        largest bucket.

    Raises:
        ValueError: If a key column is missing, the file uses a reserved column
            name, or a per-key flag list has the wrong length.
    """
    if isinstance(source, pl.LazyFrame):
        scan, label = source, "the source plan"
    else:
        scan, label = pl.scan_parquet(Path(source)), str(source)
    destination = Path(destination)
    keys = [by] if isinstance(by, str) else list(by)
    descending_flags = _per_key(descending, keys, "descending")
    nulls_last_flags = _per_key(nulls_last, keys, "nulls_last")
    if bucket_rows < 1:
        raise ValueError("bucket_rows must be at least 1")
    if sample_rows < 1:
        raise ValueError("sample_rows must be at least 1")

    schema = scan.collect_schema()
    missing = [key for key in keys if key not in schema]
    if missing:
        raise ValueError(f"sort key column(s) not in {label}: {', '.join(missing)}")
    reserved = [name for name in (ROW_INDEX_COLUMN, BUCKET_COLUMN) if name in schema]
    if reserved:
        raise ValueError(f"{label} uses reserved column name(s): {', '.join(reserved)}")

    # Parquet metadata only on a plain scan; this does not read the data.
    rows = int(scan.select(pl.len()).collect(engine="streaming").item())
    destination.parent.mkdir(parents=True, exist_ok=True)

    bucket_count = max(1, -(-rows // bucket_rows))
    if bucket_count == 1:
        scan.sort(
            keys,
            descending=descending_flags,
            nulls_last=nulls_last_flags,
            maintain_order=True,
        ).sink_parquet(destination)
        return SortReport(rows=rows, buckets=1, largest_bucket_rows=rows)

    # The row number is the last sort key, so a boundary on (key, row number)
    # can fall inside a run of equal keys without breaking the stable order.
    indexed = scan.with_row_index(ROW_INDEX_COLUMN)
    split_keys = [*keys, ROW_INDEX_COLUMN]
    split_descending = [*descending_flags, False]
    split_nulls_last = [*nulls_last_flags, False]
    boundaries = _sample_boundaries(
        scan,
        keys,
        descending_flags,
        nulls_last_flags,
        rows,
        bucket_count,
        sample_rows,
        split_ties=True,
    )
    bucket = _bucket_expr(
        split_keys,
        indexed.collect_schema(),
        split_descending,
        split_nulls_last,
        boundaries,
    )

    with tempfile.TemporaryDirectory(
        dir=work_directory or destination.parent,
        prefix=f".{destination.name}.sort-",
        ignore_cleanup_errors=True,
    ) as work:
        scatter_directory = Path(work) / "scatter"
        indexed.with_columns(bucket).sink_parquet(
            pl.PartitionBy(str(scatter_directory), key=BUCKET_COLUMN), mkdir=True
        )

        sorted_paths: list[Path] = []
        largest_bucket_rows = 0
        for index in range(len(boundaries) + 1):
            bucket_directory = scatter_directory / f"{BUCKET_COLUMN}={index}"
            if not bucket_directory.exists():
                continue
            sorted_path = Path(work) / f"sorted-{index:06d}.parquet"
            bucket_scan = pl.scan_parquet(
                bucket_directory / "*.parquet", hive_partitioning=False
            )
            largest_bucket_rows = max(
                largest_bucket_rows, int(bucket_scan.select(pl.len()).collect().item())
            )
            (
                bucket_scan.sort(
                    split_keys, descending=split_descending, nulls_last=split_nulls_last
                )
                .drop(ROW_INDEX_COLUMN, BUCKET_COLUMN)
                .sink_parquet(sorted_path)
            )
            sorted_paths.append(sorted_path)

        # Concatenating one scan per file streams the buckets through in
        # order. A single multi-file scan gives the same rows but prefetches
        # every file at once, which measured at 7 GB against under 1 GB here.
        pl.concat([pl.scan_parquet(path) for path in sorted_paths]).sink_parquet(
            destination
        )

    return SortReport(
        rows=rows, buckets=len(sorted_paths), largest_bucket_rows=largest_bucket_rows
    )


def partition_boundaries(
    source: Path | str | pl.LazyFrame,
    by: str | Sequence[str],
    *,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    bucket_rows: int = 20_000_000,
    sample_rows: int = 1_000_000,
) -> list[tuple]:
    """The key-range boundaries that split ``source`` into buckets of ~``bucket_rows``.

    The boundaries are a sorted sample of the key columns, cut at evenly spaced
    positions. Unlike :func:`sort_parquet`, which may split a run of equal keys
    across buckets by row number, these never split a key, so a key that
    covers much of the file makes its bucket that large. Rows whose
    key sorts before ``boundaries[0]`` belong to bucket 0, before
    ``boundaries[1]`` to bucket 1, and so on, so there are ``len(boundaries) + 1``
    buckets. Equal keys always share a bucket, which is what lets a per-key
    operation (a sort, a group-by, a window) run bucket by bucket in bounded
    memory. Deriving the boundaries from the *widest* input and scattering every
    input with :func:`scatter_by_boundaries` on the same list keeps a key's rows
    together across files.

    An empty list means one bucket: the file is at or under ``bucket_rows``.

    Args:
        source: The Parquet file whose key distribution sets the boundaries, or
            a lazy plan. A plan is run twice here (row count, then sample), so
            keep it to a scan plus cheap projections.
        by: The key column(s), most significant first.
        descending: Sort direction, one flag for all keys or one per key.
        nulls_last: Null placement, one flag for all keys or one per key.
        bucket_rows: Target rows per bucket.
        sample_rows: Rows sampled from the key columns to choose the boundaries.

    Returns:
        The boundary key tuples, one fewer than the number of buckets.
    """
    scan = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(Path(source))
    keys = [by] if isinstance(by, str) else list(by)
    descending_flags = _per_key(descending, keys, "descending")
    nulls_last_flags = _per_key(nulls_last, keys, "nulls_last")
    if bucket_rows < 1:
        raise ValueError("bucket_rows must be at least 1")
    if sample_rows < 1:
        raise ValueError("sample_rows must be at least 1")

    rows = int(scan.select(pl.len()).collect(engine="streaming").item())
    bucket_count = max(1, -(-rows // bucket_rows))
    if bucket_count == 1:
        return []
    return _sample_boundaries(
        scan, keys, descending_flags, nulls_last_flags, rows, bucket_count, sample_rows
    )


def scatter_by_boundaries(
    source: Path | str | pl.LazyFrame,
    directory: Path | str,
    by: str | Sequence[str],
    boundaries: Sequence[tuple],
    *,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    with_row_index: bool = True,
) -> list[Path]:
    """Route every row of ``source`` to the bucket its key falls in, on disk.

    Streams ``source`` once, writing one Parquet directory per bucket under
    ``directory`` (``__sort_bucket=<index>``). A bucket a row never reaches has
    no directory, so the returned list is only the buckets that got rows, in
    key order. Each row keeps a :data:`ROW_INDEX_COLUMN` recording its original
    position, so a reader can restore the input's order within a bucket (the
    scatter itself may reorder rows within a bucket); :data:`BUCKET_COLUMN` is
    written too. Both are reserved names and must be dropped after reading.

    Pass the ``boundaries`` from :func:`partition_boundaries` (an empty list
    routes everything to bucket 0). The same list must be used for every input
    that shares a key space, so a key's rows land in the same bucket across
    files.

    Args:
        source: The Parquet file to scatter, or a lazy plan.
        directory: Where the bucket directories are written. Created if absent.
        by: The key column(s) the boundaries are expressed in, most significant
            first.
        boundaries: The bucket boundaries, from :func:`partition_boundaries`.
        descending: Sort direction the boundaries were chosen with.
        nulls_last: Null placement the boundaries were chosen with.
        with_row_index: Stamp each row's original position as
            :data:`ROW_INDEX_COLUMN`. On by default.

    Returns:
        The bucket directories that received rows, in key order.

    Raises:
        ValueError: If ``source`` already uses a reserved column name.
    """
    scan = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(Path(source))
    keys = [by] if isinstance(by, str) else list(by)
    descending_flags = _per_key(descending, keys, "descending")
    nulls_last_flags = _per_key(nulls_last, keys, "nulls_last")
    directory = Path(directory)

    schema = scan.collect_schema()
    reserved = [name for name in (ROW_INDEX_COLUMN, BUCKET_COLUMN) if name in schema]
    if reserved:
        raise ValueError(f"source uses reserved column name(s): {', '.join(reserved)}")

    if boundaries:
        bucket = _bucket_expr(
            keys, schema, descending_flags, nulls_last_flags, list(boundaries)
        )
    else:
        bucket = pl.lit(0, dtype=pl.UInt32).alias(BUCKET_COLUMN)

    base = scan.with_row_index(ROW_INDEX_COLUMN) if with_row_index else scan
    base.with_columns(bucket).sink_parquet(
        pl.PartitionBy(str(directory), key=BUCKET_COLUMN), mkdir=True
    )

    present: list[Path] = []
    for index in range(len(boundaries) + 1):
        bucket_directory = directory / f"{BUCKET_COLUMN}={index}"
        if bucket_directory.exists():
            present.append(bucket_directory)
    return present


def read_scattered_bucket(bucket_directory: Path | str) -> pl.LazyFrame:
    """Scan one bucket of :func:`scatter_by_boundaries`, in the input's order.

    Restores the original row order the scatter recorded and drops the two
    bookkeeping columns, so the result is the input's rows for this bucket, in
    their original order, with their original columns.
    """
    return (
        pl.scan_parquet(Path(bucket_directory) / "*.parquet", hive_partitioning=False)
        .sort(ROW_INDEX_COLUMN)
        .drop(ROW_INDEX_COLUMN, BUCKET_COLUMN)
    )


def partition_reduce(
    inputs: Mapping[str, Path | str | pl.LazyFrame],
    key: str | Sequence[str],
    reduce_fn: Callable[
        [Mapping[str, pl.LazyFrame | None]], Mapping[str, pl.DataFrame]
    ],
    parts_dir: Path | str,
    *,
    widest: str,
    bucket_rows: int = 20_000_000,
    sample_rows: int = 1_000_000,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    work_directory: Path | str | None = None,
) -> dict[str, list[Path]]:
    """Run a keyed reduce one key-range bucket at a time, in bounded memory.

    This is the driver behind the "range-partition by the key, reduce per
    bucket" pattern: it range-partitions every named input on the same ``key``,
    then calls ``reduce_fn`` once per bucket with that bucket's frames and writes
    each fragment it returns to ``parts_dir`` in key order. Peak memory is one
    bucket, whatever the total size, and because the buckets are visited in key
    order the parts of any one output concatenate back in key order -- so a
    trailing global sort on the key is never needed.

    Every dangerous op the reduce performs (``.sort``, ``.over``, ``group_by``,
    ``.unique``, a big join) must be **key-local**, i.e. need only the rows that
    share a key. The boundaries come from ``inputs[widest]`` and every input is
    scattered on that same boundary list, so a key's rows co-locate across
    inputs; pass the input with the most rows as ``widest`` so its buckets bound
    every co-partitioned input.

    Args:
        inputs: Named sources to co-partition. A source is a Parquet file or a
            cheap lazy scan (it is streamed once by the scatter). Materialize any
            expensive plan -- one containing a join or ``.over`` -- to a file
            first, because :func:`partition_boundaries` re-runs ``widest``.
        key: The partition key column(s), most significant first.
        reduce_fn: Called once per non-empty bucket with ``{name: LazyFrame}``,
            where a name maps to ``None`` when that input had no rows in the
            bucket. It returns ``{output_name: DataFrame}``, the bucket's
            fragment of each output. The same output names must be returned every
            call. When every input is empty it is called once with all-``None``
            so the outputs still get an (empty) part that fixes their schema.
        parts_dir: Where the output parts are written, one subdirectory per
            output name, one ``NNNNNN.parquet`` per bucket. The caller owns this
            directory and concatenates the returned parts.
        widest: The ``inputs`` key whose distribution sets the bucket
            boundaries; make it the input with the most rows.
        bucket_rows: Target rows per bucket, bounding the per-bucket memory.
        sample_rows: Rows sampled from ``widest`` to choose the boundaries.
        descending: Sort direction of the key, one flag for all or one per key.
        nulls_last: Null placement of the key, one flag for all or one per key.
        work_directory: Where the scattered buckets live during the reduce.
            Defaults to ``parts_dir``'s parent, so the work lands on the same
            disk. Must have room for a copy of every input.

    Returns:
        The ordered part paths for each output name, in key order.

    Raises:
        ValueError: If ``widest`` is not one of ``inputs``.
    """
    if widest not in inputs:
        raise ValueError(f"widest {widest!r} is not one of the inputs {sorted(inputs)}")
    parts_dir = Path(parts_dir)
    keys = [key] if isinstance(key, str) else list(key)

    boundaries = partition_boundaries(
        inputs[widest],
        keys,
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=bucket_rows,
        sample_rows=sample_rows,
    )

    ordered: dict[str, list[Path]] = {}

    def _write(index: int, fragments: Mapping[str, pl.DataFrame]) -> None:
        for name, frame in fragments.items():
            directory = parts_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{index:06d}.parquet"
            frame.write_parquet(path)
            ordered.setdefault(name, []).append(path)

    with tempfile.TemporaryDirectory(
        dir=work_directory or parts_dir.parent,
        prefix=".partition-reduce-",
        ignore_cleanup_errors=True,
    ) as work:
        scatter_root = Path(work)
        scatter_dirs = {}
        for name, source in inputs.items():
            directory = scatter_root / name
            scatter_by_boundaries(
                source,
                directory,
                keys,
                boundaries,
                descending=descending,
                nulls_last=nulls_last,
            )
            scatter_dirs[name] = directory

        produced = False
        for index in range(len(boundaries) + 1):
            frames: dict[str, pl.LazyFrame | None] = {}
            present = False
            for name, directory in scatter_dirs.items():
                bucket_directory = directory / f"{BUCKET_COLUMN}={index}"
                if bucket_directory.exists():
                    frames[name] = read_scattered_bucket(bucket_directory)
                    present = True
                else:
                    frames[name] = None
            if not present:
                continue
            _write(index, reduce_fn(frames))
            produced = True

        if not produced:
            _write(0, reduce_fn({name: None for name in inputs}))

    return ordered


def distinct_count(
    source: Path | str | pl.LazyFrame,
    by: str | Sequence[str],
    *,
    bucket_rows: int = 20_000_000,
    sample_rows: int = 1_000_000,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    work_directory: Path | str | None = None,
) -> int:
    """The number of distinct ``by`` values in ``source``, in bounded memory.

    Equal to ``pl.scan_parquet(source).select(by).unique().collect().height`` and
    to ``pl.col(by).n_unique()``, but never holds the whole distinct set in RAM:
    it range-partitions on ``by`` so equal keys share a bucket, dedupes one
    bucket at a time, and sums the per-bucket distinct counts. Peak memory is one
    bucket's distinct set, whatever the total cardinality -- which is what
    ``n_unique`` cannot promise on a population-scale column.

    Args:
        source: The Parquet file to count over, or a cheap lazy scan. A plan is
            run up to three times (row count, sample, scatter), so keep it to a
            scan plus projection.
        by: The column(s) whose distinct combinations are counted.
        bucket_rows: Target rows per bucket, bounding the per-bucket memory.
        sample_rows: Rows sampled from ``by`` to choose the bucket boundaries.
        descending: Sort direction of the key, one flag for all or one per key.
        nulls_last: Null placement of the key, one flag for all or one per key.
        work_directory: Where the scattered buckets live. Defaults to the system
            temporary directory. Must have room for a copy of the key columns.

    Returns:
        The count of distinct ``by`` values.
    """
    scan = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(Path(source))
    keys = [by] if isinstance(by, str) else list(by)
    projected = scan.select(keys)

    boundaries = partition_boundaries(
        projected,
        keys,
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=bucket_rows,
        sample_rows=sample_rows,
    )

    with tempfile.TemporaryDirectory(
        dir=work_directory, prefix=".distinct-count-", ignore_cleanup_errors=True
    ) as work:
        directory = Path(work)
        # The row index is not needed -- distinctness does not care about order.
        scatter_by_boundaries(
            projected,
            directory,
            keys,
            boundaries,
            descending=descending,
            nulls_last=nulls_last,
            with_row_index=False,
        )
        total = 0
        for index in range(len(boundaries) + 1):
            bucket_directory = directory / f"{BUCKET_COLUMN}={index}"
            if not bucket_directory.exists():
                continue
            bucket = pl.scan_parquet(
                bucket_directory / "*.parquet", hive_partitioning=False
            ).select(keys)
            total += int(
                bucket.unique().select(pl.len()).collect(engine="streaming").item()
            )
    return total


def distinct_parquet(
    source: Path | str | pl.LazyFrame,
    destination: Path | str,
    by: str | Sequence[str],
    *,
    bucket_rows: int = 20_000_000,
    sample_rows: int = 1_000_000,
    work_directory: Path | str | None = None,
) -> int:
    """Write the rows of ``source`` de-duplicated on ``by`` to ``destination``.

    Equal to ``scan(source).unique(subset=by).sink_parquet(destination)`` but
    never holds the whole distinct set in RAM: it range-partitions on ``by`` so
    equal keys share a bucket, dedupes one bucket at a time, and concatenates the
    buckets. A file at or under ``bucket_rows`` is deduped in one in-memory step,
    so that is the "fits in memory -> just call ``.unique()``" branch. Row order
    is not preserved -- a dedup has none to preserve.

    Args:
        source: The Parquet file to dedupe, or a cheap lazy scan. A plan is run
            up to three times (row count, sample, scatter), so materialize any
            expensive plan first.
        destination: Where to write the deduped file. Its parent is created.
        by: The column(s) to dedupe on. Rows sharing these values collapse to
            one, keeping every column, exactly like ``.unique(subset=by)``.
        bucket_rows: Target rows per bucket, bounding the per-bucket memory.
        sample_rows: Rows sampled from ``by`` to choose the bucket boundaries.
        work_directory: Where the scattered buckets live. Defaults to the
            destination's directory.

    Returns:
        The number of distinct rows written.
    """
    scan = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(Path(source))
    keys = [by] if isinstance(by, str) else list(by)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    boundaries = partition_boundaries(
        scan, keys, bucket_rows=bucket_rows, sample_rows=sample_rows
    )
    if not boundaries:
        deduped = scan.unique(subset=keys).collect(engine="streaming")
        deduped.write_parquet(destination)
        return deduped.height

    with tempfile.TemporaryDirectory(
        dir=work_directory or destination.parent,
        prefix=f".{destination.name}.distinct-",
        ignore_cleanup_errors=True,
    ) as work:
        scatter_directory = Path(work) / "scatter"
        scatter_by_boundaries(
            scan, scatter_directory, keys, boundaries, with_row_index=False
        )
        part_paths: list[Path] = []
        total = 0
        for index in range(len(boundaries) + 1):
            bucket_directory = scatter_directory / f"{BUCKET_COLUMN}={index}"
            if not bucket_directory.exists():
                continue
            deduped = (
                pl.scan_parquet(bucket_directory / "*.parquet", hive_partitioning=False)
                .drop(BUCKET_COLUMN)
                .unique(subset=keys)
                .collect(engine="streaming")
            )
            part_path = Path(work) / f"distinct-{index:06d}.parquet"
            deduped.write_parquet(part_path)
            part_paths.append(part_path)
            total += deduped.height
        pl.concat([pl.scan_parquet(path) for path in part_paths]).sink_parquet(
            destination
        )
    return total


def _per_key(flag: bool | Sequence[bool], keys: list[str], name: str) -> list[bool]:
    if isinstance(flag, bool):
        return [flag] * len(keys)
    flags = list(flag)
    if len(flags) != len(keys):
        raise ValueError(
            f"{name} has {len(flags)} value(s) for {len(keys)} sort key column(s)"
        )
    return flags


def _sample_boundaries(
    scan: pl.LazyFrame,
    keys: list[str],
    descending: list[bool],
    nulls_last: list[bool],
    rows: int,
    bucket_count: int,
    sample_rows: int,
    *,
    split_ties: bool = False,
) -> list[tuple]:
    """Evenly spaced key values from a sorted sample, one fewer than the buckets.

    Only the key columns are read, and only every ``step``-th row is kept, so
    the sample is small however large the file is. Consecutive equal boundaries
    are dropped: a repeated boundary only yields an empty bucket.

    With ``split_ties`` each boundary also carries the sampled row's
    :data:`ROW_INDEX_COLUMN` as a final ascending key. Boundaries are then
    unique, so none are dropped and a repeated key is spread over buckets,
    but the buckets must be assigned on ``(*keys, ROW_INDEX_COLUMN)`` to match.
    """
    step = max(1, rows // sample_rows)
    sampled = (
        scan.select(keys)
        .with_row_index(ROW_INDEX_COLUMN)
        .filter(pl.col(ROW_INDEX_COLUMN) % step == 0)
    )
    if split_ties:
        # Key columns first, so each boundary tuple lines up with the keys.
        sample = sampled.select(*keys, ROW_INDEX_COLUMN).sort(
            [*keys, ROW_INDEX_COLUMN],
            descending=[*descending, False],
            nulls_last=[*nulls_last, False],
        )
    else:
        sample = sampled.drop(ROW_INDEX_COLUMN).sort(
            keys, descending=descending, nulls_last=nulls_last
        )
    sample = sample.collect(engine="streaming")
    boundaries: list[tuple] = []
    for index in range(1, bucket_count):
        boundary = sample.row(index * sample.height // bucket_count)
        if not boundaries or boundary != boundaries[-1]:
            boundaries.append(boundary)
    return boundaries


def _bucket_expr(
    keys: list[str],
    schema: pl.Schema,
    descending: list[bool],
    nulls_last: list[bool],
    boundaries: list[tuple],
) -> pl.Expr:
    """The bucket of each row: how many boundaries sort at or before its key."""
    at_or_after = [
        _at_or_after(keys, schema, descending, nulls_last, boundary).cast(pl.UInt32)
        for boundary in boundaries
    ]
    return pl.sum_horizontal(at_or_after).alias(BUCKET_COLUMN)


def _at_or_after(
    keys: list[str],
    schema: pl.Schema,
    descending: list[bool],
    nulls_last: list[bool],
    boundary: tuple,
) -> pl.Expr:
    """Whether a row's key sorts at or after ``boundary``.

    Lexicographic over the keys: the row is at or after the boundary when its
    first key is after the boundary's, or equal and the remaining keys are at
    or after the rest of the boundary.
    """
    result = pl.lit(True)
    per_key = zip(keys, descending, nulls_last, boundary, strict=True)
    for key, is_descending, is_nulls_last, value in reversed(list(per_key)):
        after, same = _compare(
            pl.col(key), schema[key], is_descending, is_nulls_last, value
        )
        result = after | (same & result)
    return result


def _compare(
    column: pl.Expr,
    dtype: pl.DataType,
    descending: bool,
    nulls_last: bool,
    value: object,
) -> tuple[pl.Expr, pl.Expr]:
    """``(column sorts after value, column equals value)``, both never null.

    Polars orders NaN above every other float in comparisons as in sorts, so
    float keys need no special case.
    """
    if value is None:
        # Null sorts before everything unless nulls_last, so nothing sorts
        # after a null boundary in that case, and every non-null does otherwise.
        after = pl.lit(False) if nulls_last else column.is_not_null()
        return after, column.is_null()
    literal = pl.lit(value, dtype=dtype)
    after = column < literal if descending else column > literal
    # A null row sorts before every value, or after all of them, by nulls_last.
    return after.fill_null(nulls_last), (column == literal).fill_null(False)
