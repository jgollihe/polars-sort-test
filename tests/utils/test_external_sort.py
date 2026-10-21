"""Unit tests for the range-partitioned external Parquet sort."""

import datetime as dt
import random
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from utils.external_sort import (
    BUCKET_COLUMN,
    ROW_INDEX_COLUMN,
    SortReport,
    distinct_count,
    distinct_parquet,
    partition_boundaries,
    partition_reduce,
    read_scattered_bucket,
    scatter_by_boundaries,
    sort_parquet,
)

NAN = float("nan")


def _expected(frame: pl.DataFrame, by, **kwargs) -> pl.DataFrame:
    """The in-memory stable sort the external sort must reproduce."""
    return frame.sort(by, maintain_order=True, **kwargs)


def _sort(
    tmp_path: Path, frame: pl.DataFrame, by, **kwargs
) -> tuple[pl.DataFrame, SortReport]:
    source = tmp_path / "input.parquet"
    destination = tmp_path / "out" / "sorted.parquet"
    frame.write_parquet(source)
    report = sort_parquet(source, destination, by, **kwargs)
    return pl.read_parquet(destination), report


def _random_frame(seed: int, height: int) -> pl.DataFrame:
    """Heavily duplicated keys of several types, with nulls and NaN, plus a row id."""
    rng = random.Random(seed)
    words = ["ANN", "BOB", "CAL", "DEE", "", "ann"]
    return pl.DataFrame(
        {
            "int_key": [rng.choice([None, *range(6)]) for _ in range(height)],
            "str_key": [rng.choice([None, *words]) for _ in range(height)],
            "float_key": [
                rng.choice([None, NAN, 0.5, 1.5, -2.0]) for _ in range(height)
            ],
            "date_key": [
                rng.choice([None, dt.date(2020, 1, 1), dt.date(2021, 6, 30)])
                for _ in range(height)
            ],
            "row": list(range(height)),
        }
    )


def test_single_bucket_matches_in_memory_stable_sort(tmp_path: Path) -> None:
    frame = _random_frame(seed=1, height=100)
    result, report = _sort(tmp_path, frame, ["int_key", "str_key"])
    assert_frame_equal(result, _expected(frame, ["int_key", "str_key"]))
    assert report == SortReport(rows=100, buckets=1, largest_bucket_rows=100)


def test_many_buckets_match_in_memory_stable_sort(tmp_path: Path) -> None:
    frame = _random_frame(seed=2, height=500)
    result, report = _sort(tmp_path, frame, "int_key", bucket_rows=20, sample_rows=100)
    assert_frame_equal(result, _expected(frame, "int_key"))
    assert report.rows == 500
    assert report.buckets > 1


@pytest.mark.parametrize("key", ["int_key", "str_key", "float_key", "date_key"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_every_key_type_direction_and_null_placement(
    tmp_path: Path, key: str, descending: bool, nulls_last: bool
) -> None:
    frame = _random_frame(seed=3, height=300)
    result, report = _sort(
        tmp_path,
        frame,
        key,
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=25,
        sample_rows=60,
    )
    expected = _expected(frame, key, descending=descending, nulls_last=nulls_last)
    assert_frame_equal(result, expected)
    assert report.buckets > 1


def test_multi_column_key_with_per_column_flags(tmp_path: Path) -> None:
    frame = _random_frame(seed=4, height=400)
    by = ["str_key", "float_key", "int_key", "date_key"]
    descending = [True, False, True, False]
    nulls_last = [False, True, True, False]
    result, report = _sort(
        tmp_path,
        frame,
        by,
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=30,
        sample_rows=80,
    )
    expected = _expected(frame, by, descending=descending, nulls_last=nulls_last)
    assert_frame_equal(result, expected)
    assert report.buckets > 1


@pytest.mark.parametrize("seed", range(5))
def test_randomized_against_in_memory_sort(tmp_path: Path, seed: int) -> None:
    rng = random.Random(100 + seed)
    frame = _random_frame(seed=seed, height=rng.randint(50, 600))
    keys = ["int_key", "str_key", "float_key", "date_key"]
    by = rng.sample(keys, rng.randint(1, len(keys)))
    descending = [rng.random() < 0.5 for _ in by]
    nulls_last = [rng.random() < 0.5 for _ in by]
    result, _ = _sort(
        tmp_path,
        frame,
        by,
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=rng.randint(1, 40),
        sample_rows=rng.randint(1, 200),
    )
    expected = _expected(frame, by, descending=descending, nulls_last=nulls_last)
    assert_frame_equal(result, expected)


def test_equal_keys_keep_input_order_across_buckets(tmp_path: Path) -> None:
    # Ties are split across buckets by row number, which is also the final
    # sort key within a bucket, so the result is still the stable sort.
    frame = pl.DataFrame({"key": [1, 0] * 200, "row": list(range(400))})
    result, report = _sort(tmp_path, frame, "key", bucket_rows=10, sample_rows=1000)
    assert_frame_equal(result, _expected(frame, "key"))
    assert report.buckets == 40
    assert report.largest_bucket_rows == 10


def test_one_heavy_key_is_split_to_the_bucket_target(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"key": ["B"] * 300 + ["A", "C"] * 10, "row": list(range(320))}
    )
    result, report = _sort(tmp_path, frame, "key", bucket_rows=8, sample_rows=1000)
    assert_frame_equal(result, _expected(frame, "key"))
    # A full sample gives exact boundaries, so every bucket is the target size
    # even though one key covers 300 of the 320 rows.
    assert report.buckets == 40
    assert report.largest_bucket_rows == 8


def test_single_key_value_is_split_to_the_bucket_target(tmp_path: Path) -> None:
    frame = pl.DataFrame({"key": ["same"] * 1000, "row": list(range(1000))})
    result, report = _sort(tmp_path, frame, "key", bucket_rows=100, sample_rows=1000)
    assert_frame_equal(result, frame)
    assert report.buckets == 10
    assert report.largest_bucket_rows == 100


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_heavy_null_and_value_runs_are_split_and_stay_stable(
    tmp_path: Path, descending: bool, nulls_last: bool
) -> None:
    # Two long runs, one of them null, interleaved so each run's input order
    # must be restored across the buckets it is split over.
    frame = pl.DataFrame(
        {"key": [None, 7] * 300 + [3] * 20, "row": list(range(620))}
    )
    result, report = _sort(
        tmp_path,
        frame,
        "key",
        descending=descending,
        nulls_last=nulls_last,
        bucket_rows=20,
        sample_rows=1000,
    )
    expected = _expected(frame, "key", descending=descending, nulls_last=nulls_last)
    assert_frame_equal(result, expected)
    assert report.largest_bucket_rows <= 20


@pytest.mark.parametrize("seed", range(5))
def test_sparse_sample_of_few_keys_keeps_buckets_near_target(
    tmp_path: Path, seed: int
) -> None:
    # A sample of every 10th row only estimates the boundaries, so buckets are
    # near the target rather than exact, but nowhere near a whole key's run.
    rng = random.Random(200 + seed)
    height = 4000
    frame = pl.DataFrame(
        {
            "key": [rng.choice(["AIR", "RAIL", "SHIP", "MAIL"]) for _ in range(height)],
            "row": list(range(height)),
        }
    )
    result, report = _sort(tmp_path, frame, "key", bucket_rows=200, sample_rows=400)
    assert_frame_equal(result, _expected(frame, "key"))
    assert report.buckets == 20
    assert report.largest_bucket_rows <= 2 * 200


def test_multi_column_key_with_few_combinations_is_split(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "a": ["x", "y"] * 250,
            "b": [1, 1, 2, 2] * 125,
            "row": list(range(500)),
        }
    )
    by = ["a", "b"]
    result, report = _sort(
        tmp_path,
        frame,
        by,
        descending=[True, False],
        bucket_rows=25,
        sample_rows=1000,
    )
    assert_frame_equal(result, _expected(frame, by, descending=[True, False]))
    assert report.buckets == 20
    assert report.largest_bucket_rows == 25


def test_tiny_sample_still_sorts_correctly(tmp_path: Path) -> None:
    frame = _random_frame(seed=5, height=250)
    result, _ = _sort(tmp_path, frame, "int_key", bucket_rows=5, sample_rows=1)
    assert_frame_equal(result, _expected(frame, "int_key"))


def test_all_null_key(tmp_path: Path) -> None:
    frame = pl.DataFrame({"key": [None] * 50, "row": list(range(50))}).with_columns(
        pl.col("key").cast(pl.Int64)
    )
    result, _ = _sort(tmp_path, frame, "key", bucket_rows=7, sample_rows=10)
    assert_frame_equal(result, frame)


def test_empty_file(tmp_path: Path) -> None:
    frame = pl.DataFrame({"key": pl.Series([], dtype=pl.Int64), "row": []})
    result, report = _sort(tmp_path, frame, "key", bucket_rows=1)
    assert_frame_equal(result, frame)
    assert report == SortReport(rows=0, buckets=1, largest_bucket_rows=0)


def test_output_is_one_file_and_work_directory_is_removed(tmp_path: Path) -> None:
    frame = _random_frame(seed=6, height=200)
    _sort(tmp_path, frame, "int_key", bucket_rows=10, sample_rows=50)
    output_directory = tmp_path / "out"
    assert [p.name for p in output_directory.iterdir()] == ["sorted.parquet"]


def test_explicit_work_directory_is_used_and_cleaned(tmp_path: Path) -> None:
    frame = _random_frame(seed=7, height=200)
    work = tmp_path / "scratch"
    work.mkdir()
    result, _ = _sort(
        tmp_path, frame, "int_key", bucket_rows=10, sample_rows=50, work_directory=work
    )
    assert_frame_equal(result, _expected(frame, "int_key"))
    assert list(work.iterdir()) == []


def test_helper_columns_do_not_leak_into_the_output(tmp_path: Path) -> None:
    frame = _random_frame(seed=8, height=120)
    result, _ = _sort(tmp_path, frame, "int_key", bucket_rows=10, sample_rows=50)
    assert result.columns == frame.columns
    assert result.schema == frame.schema


def test_missing_key_column_is_rejected(tmp_path: Path) -> None:
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "input.parquet")
    with pytest.raises(ValueError, match="not in .*: b, c"):
        sort_parquet(
            tmp_path / "input.parquet", tmp_path / "out.parquet", ["a", "b", "c"]
        )


@pytest.mark.parametrize("reserved", [ROW_INDEX_COLUMN, BUCKET_COLUMN])
def test_reserved_column_names_are_rejected(tmp_path: Path, reserved: str) -> None:
    pl.DataFrame({"a": [1], reserved: [1]}).write_parquet(tmp_path / "input.parquet")
    with pytest.raises(ValueError, match="reserved"):
        sort_parquet(tmp_path / "input.parquet", tmp_path / "out.parquet", "a")


@pytest.mark.parametrize("flag", ["descending", "nulls_last"])
def test_per_key_flag_length_must_match(tmp_path: Path, flag: str) -> None:
    pl.DataFrame({"a": [1], "b": [2]}).write_parquet(tmp_path / "input.parquet")
    with pytest.raises(ValueError, match=f"{flag} has 3 value"):
        sort_parquet(
            tmp_path / "input.parquet",
            tmp_path / "out.parquet",
            ["a", "b"],
            **{flag: [True, False, True]},
        )


@pytest.mark.parametrize("argument", ["bucket_rows", "sample_rows"])
def test_sizes_must_be_positive(tmp_path: Path, argument: str) -> None:
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "input.parquet")
    with pytest.raises(ValueError, match=argument):
        sort_parquet(
            tmp_path / "input.parquet", tmp_path / "out.parquet", "a", **{argument: 0}
        )


def test_lazy_plan_source(tmp_path: Path) -> None:
    frame = _random_frame(seed=9, height=300)
    frame.write_parquet(tmp_path / "input.parquet")
    plan = pl.scan_parquet(tmp_path / "input.parquet").filter(pl.col("row") % 2 == 0)
    report = sort_parquet(
        plan, tmp_path / "sorted.parquet", "str_key", bucket_rows=20, sample_rows=50
    )
    expected = _expected(frame.filter(pl.col("row") % 2 == 0), "str_key")
    assert_frame_equal(pl.read_parquet(tmp_path / "sorted.parquet"), expected)
    assert report.rows == 150
    assert report.buckets > 1


def test_lazy_plan_source_missing_key_names_the_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the source plan: b"):
        sort_parquet(pl.LazyFrame({"a": [1]}), tmp_path / "out.parquet", "b")


def _reassemble(directories: list[Path]) -> pl.DataFrame:
    """Read every bucket in order, as a partitioned consumer would."""
    return pl.concat(
        [read_scattered_bucket(directory) for directory in directories]
    ).collect()


def test_partition_boundaries_are_empty_when_one_bucket_suffices() -> None:
    frame = _random_frame(seed=10, height=50)
    assert partition_boundaries(frame.lazy(), "int_key", bucket_rows=1000) == []


def test_scatter_colocates_equal_keys_and_preserves_row_order(tmp_path: Path) -> None:
    frame = _random_frame(seed=11, height=400)
    frame.write_parquet(tmp_path / "input.parquet")
    boundaries = partition_boundaries(
        tmp_path / "input.parquet", "int_key", bucket_rows=40, sample_rows=100
    )
    directories = scatter_by_boundaries(
        tmp_path / "input.parquet", tmp_path / "scatter", "int_key", boundaries
    )
    assert len(directories) > 1

    # Every row is present exactly once, and each key lives in exactly one bucket.
    seen_keys: set = set()
    for directory in directories:
        bucket = read_scattered_bucket(directory).collect()
        keys = set(bucket["int_key"].to_list())
        assert keys.isdisjoint(seen_keys)
        seen_keys |= keys
        # Within a bucket the input's relative order is restored.
        assert bucket["row"].to_list() == sorted(bucket["row"].to_list())
    assert _reassemble(directories).sort("row").equals(frame.sort("row"))


def test_scatter_with_empty_boundaries_writes_one_bucket(tmp_path: Path) -> None:
    frame = _random_frame(seed=12, height=30)
    frame.write_parquet(tmp_path / "input.parquet")
    directories = scatter_by_boundaries(
        tmp_path / "input.parquet", tmp_path / "scatter", "int_key", []
    )
    assert len(directories) == 1
    assert_frame_equal(_reassemble(directories), frame)


@pytest.mark.parametrize("reserved", [ROW_INDEX_COLUMN, BUCKET_COLUMN])
def test_scatter_rejects_reserved_column_names(tmp_path: Path, reserved: str) -> None:
    frame = pl.DataFrame({"int_key": [1, 2], reserved: [1, 2]})
    with pytest.raises(ValueError, match="reserved"):
        scatter_by_boundaries(frame.lazy(), tmp_path / "scatter", "int_key", [])


# ---------------------------------------------------------------------------
# partition_reduce
# ---------------------------------------------------------------------------


def _grouped_sum(frames):
    """A key-local reduce: sum ``value`` per ``key`` within the bucket."""
    frame = frames["data"]
    if frame is None:
        frame = pl.LazyFrame(schema={"key": pl.Int64, "value": pl.Int64})
    return {"sums": frame.group_by("key").agg(pl.col("value").sum()).collect()}


@pytest.mark.parametrize("bucket_rows", [1, 3, 1000])
def test_partition_reduce_matches_a_single_group_by(tmp_path: Path, bucket_rows: int):
    rng = random.Random(7)
    frame = pl.DataFrame(
        {
            "key": [rng.randint(0, 40) for _ in range(400)],
            "value": [rng.randint(0, 9) for _ in range(400)],
        }
    )
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)

    parts = partition_reduce(
        {"data": source},
        "key",
        _grouped_sum,
        tmp_path / "parts",
        widest="data",
        bucket_rows=bucket_rows,
        sample_rows=1000,
    )
    got = pl.concat([pl.read_parquet(p) for p in parts["sums"]]).sort("key")
    want = frame.group_by("key").agg(pl.col("value").sum()).sort("key")
    assert_frame_equal(got, want)


def test_partition_reduce_concatenates_buckets_in_key_order(tmp_path: Path):
    """When the reduce sorts within a bucket, the concatenation is globally sorted."""
    rng = random.Random(9)
    frame = pl.DataFrame({"key": [rng.randint(0, 40) for _ in range(400)]})
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)

    def sorted_bucket(frames):
        bucket = frames["data"]
        if bucket is None:
            bucket = pl.LazyFrame(schema={"key": pl.Int64})
        return {"out": bucket.sort("key").collect()}

    parts = partition_reduce(
        {"data": source},
        "key",
        sorted_bucket,
        tmp_path / "parts",
        widest="data",
        bucket_rows=20,
        sample_rows=400,
    )
    assert len(parts["out"]) > 1  # actually partitioned
    keys = pl.concat([pl.read_parquet(p) for p in parts["out"]])["key"].to_list()
    assert keys == sorted(keys)


def test_partition_reduce_runs_once_on_all_empty_inputs(tmp_path: Path):
    source = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"key": pl.Int64, "value": pl.Int64}).write_parquet(source)
    parts = partition_reduce(
        {"data": source}, "key", _grouped_sum, tmp_path / "parts", widest="data"
    )
    assert len(parts["sums"]) == 1
    assert pl.read_parquet(parts["sums"][0]).is_empty()


def test_partition_reduce_rejects_unknown_widest(tmp_path: Path):
    source = tmp_path / "data.parquet"
    pl.DataFrame({"key": [1], "value": [1]}).write_parquet(source)
    with pytest.raises(ValueError, match="widest"):
        partition_reduce(
            {"data": source}, "key", _grouped_sum, tmp_path / "parts", widest="other"
        )


# ---------------------------------------------------------------------------
# distinct_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket_rows", [1, 2, 7, 1000])
def test_distinct_count_matches_n_unique(tmp_path: Path, bucket_rows: int):
    frame = _random_frame(seed=11, height=600)
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)

    want = int(frame.select(pl.col("str_key").n_unique()).item())
    got = distinct_count(source, "str_key", bucket_rows=bucket_rows, sample_rows=200)
    assert got == want


def test_distinct_count_over_multiple_keys(tmp_path: Path):
    frame = _random_frame(seed=12, height=600)
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)

    want = frame.select("int_key", "str_key").unique().height
    got = distinct_count(source, ["int_key", "str_key"], bucket_rows=5, sample_rows=200)
    assert got == want


def test_distinct_count_accepts_a_lazy_plan(tmp_path: Path):
    frame = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [9, 8, 7, 6, 5]})
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)
    assert distinct_count(pl.scan_parquet(source).select("a"), "a", bucket_rows=1) == 3


# ---------------------------------------------------------------------------
# distinct_parquet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket_rows", [1, 2, 7, 1000])
def test_distinct_parquet_matches_in_memory_unique(tmp_path: Path, bucket_rows: int):
    frame = _random_frame(seed=21, height=600)
    source = tmp_path / "data.parquet"
    frame.write_parquet(source)
    destination = tmp_path / "out" / "distinct.parquet"

    rows = distinct_parquet(
        source, destination, "str_key", bucket_rows=bucket_rows, sample_rows=200
    )
    got = pl.read_parquet(destination)
    want = frame.unique(subset=["str_key"])
    # Row order is not preserved by a dedup; compare as sets of the key column.
    assert rows == want.height == got.height
    assert set(got["str_key"].to_list()) == set(want["str_key"].to_list())


def test_distinct_parquet_dedupes_whole_single_column(tmp_path: Path):
    """The engine's use: dedupe a one-column id file, so unique == unique(subset)."""
    source = tmp_path / "ids.parquet"
    pl.DataFrame({"id": [3, 1, 3, 2, 1, 3, 2]}).write_parquet(source)
    destination = tmp_path / "out.parquet"
    rows = distinct_parquet(source, destination, "id", bucket_rows=2, sample_rows=100)
    assert rows == 3
    assert sorted(pl.read_parquet(destination)["id"].to_list()) == [1, 2, 3]


def test_distinct_parquet_single_bucket_fast_path(tmp_path: Path):
    source = tmp_path / "ids.parquet"
    pl.DataFrame({"id": [1, 1, 2]}).write_parquet(source)
    destination = tmp_path / "out.parquet"
    # bucket_rows above the height takes the in-memory branch, no scatter.
    rows = distinct_parquet(source, destination, "id", bucket_rows=1000)
    assert rows == 2
    assert sorted(pl.read_parquet(destination)["id"].to_list()) == [1, 2]
