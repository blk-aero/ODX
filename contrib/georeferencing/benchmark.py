#!/usr/bin/env python3
"""Validate and summarize reproducible georeferencing benchmark evidence."""

import argparse
import json
import math
import pathlib
import statistics
import sys


SCHEMA = "odx-georeferencing-benchmark-v1"
REQUIRED_BENCHMARK_CASES = (
    "point_cloud_1m",
    "point_cloud_10m",
    "mesh_100k",
    "mesh_1m",
    "raster_25mp",
    "raster_100mp",
    "boundary_ordinary",
    "boundary_forced_subdivision",
    "gcp_end_to_end",
    "gps_end_to_end",
)
PERFORMANCE_THRESHOLDS = {
    "maximum_scale_factor": 2.5,
    "maximum_total_job_regression": 0.10,
    "maximum_resource_growth": 0.25,
}
REQUIRED_FIELDS = {
    "schema", "fixture_id", "case", "variant", "command",
    "container_digest", "warmup", "iteration", "elapsed_seconds",
    "peak_rss_bytes", "output_size_bytes", "coordinate_count",
    "throughput_per_second",
}


def validate_benchmark_record(record):
    """Validate one raw benchmark measurement without inventing evidence."""
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError("benchmark record is missing: {}".format(
            ", ".join(sorted(missing))
        ))
    if record["schema"] != SCHEMA:
        raise ValueError("unsupported benchmark schema")
    if record["case"] not in REQUIRED_BENCHMARK_CASES:
        raise ValueError("unknown benchmark case")
    if record["variant"] not in ("stock", "corrected"):
        raise ValueError("variant must be stock or corrected")
    if not isinstance(record["command"], list) or not record["command"]:
        raise ValueError("command must be a non-empty argument array")
    for name in ("fixture_id", "container_digest"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError("{} must be a non-empty string".format(name))
    if not isinstance(record["warmup"], bool):
        raise ValueError("warmup must be boolean")
    for name in (
        "iteration", "elapsed_seconds", "peak_rss_bytes", "output_size_bytes",
        "coordinate_count", "throughput_per_second",
    ):
        value = record[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and non-negative".format(name))
    if record["elapsed_seconds"] == 0 or record["coordinate_count"] == 0:
        raise ValueError("elapsed time and coordinate count must be positive")


def summarize(records):
    """Return medians and release-threshold failures from raw alternating runs."""
    for record in records:
        validate_benchmark_record(record)
    measured = [record for record in records if not record["warmup"]]
    warmups = [record for record in records if record["warmup"]]
    cases = {record["case"] for record in records}
    missing_cases = set(REQUIRED_BENCHMARK_CASES) - cases
    if missing_cases:
        raise ValueError("benchmark evidence is missing cases: {}".format(
            ", ".join(sorted(missing_cases))
        ))
    summary = {"schema": SCHEMA, "thresholds": PERFORMANCE_THRESHOLDS, "cases": {}}
    failures = []
    for case in REQUIRED_BENCHMARK_CASES:
        case_records = [record for record in measured if record["case"] == case]
        case_warmups = [record for record in warmups if record["case"] == case]
        for variant in ("stock", "corrected"):
            if len([record for record in case_warmups if record["variant"] == variant]) < 1:
                raise ValueError("{} needs one {} warm-up".format(case, variant))
            if len([record for record in case_records if record["variant"] == variant]) < 3:
                raise ValueError("{} needs three measured {} runs".format(case, variant))
        variants = [record["variant"] for record in case_records]
        if any(left == right for left, right in zip(variants, variants[1:])):
            raise ValueError("{} measurements must alternate stock/corrected".format(case))
        medians = {}
        for variant in ("stock", "corrected"):
            selected = [record for record in case_records if record["variant"] == variant]
            medians[variant] = {
                field: statistics.median(record[field] for record in selected)
                for field in ("elapsed_seconds", "peak_rss_bytes", "output_size_bytes", "throughput_per_second")
            }
        summary["cases"][case] = medians
        if case.endswith("end_to_end"):
            regression = medians["corrected"]["elapsed_seconds"] / medians["stock"]["elapsed_seconds"] - 1.0
            if regression > PERFORMANCE_THRESHOLDS["maximum_total_job_regression"]:
                failures.append("{} total-job regression {:.1%}".format(case, regression))
        for field in ("peak_rss_bytes", "output_size_bytes"):
            stock = medians["stock"][field]
            if stock and medians["corrected"][field] / stock - 1.0 > PERFORMANCE_THRESHOLDS["maximum_resource_growth"]:
                failures.append("{} {} growth exceeds 25%".format(case, field))

    for small, large in (
        ("point_cloud_1m", "point_cloud_10m"),
        ("mesh_100k", "mesh_1m"),
        ("raster_25mp", "raster_100mp"),
    ):
        small_case = summary["cases"][small]["corrected"]
        large_case = summary["cases"][large]["corrected"]
        small_count = next(
            record["coordinate_count"] for record in measured
            if record["case"] == small and record["variant"] == "corrected"
        )
        large_count = next(
            record["coordinate_count"] for record in measured
            if record["case"] == large and record["variant"] == "corrected"
        )
        count_ratio = large_count / small_count
        elapsed_ratio = large_case["elapsed_seconds"] / small_case["elapsed_seconds"]
        maximum_elapsed_ratio = PERFORMANCE_THRESHOLDS[
            "maximum_scale_factor"
        ] ** math.log2(count_ratio)
        if elapsed_ratio > maximum_elapsed_ratio:
            failures.append("{} to {} scaling exceeds 2.5x per doubling".format(small, large))
    summary["failures"] = failures
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    with args.records.open() as source:
        records = json.load(source)
    summary = summarize(records)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
