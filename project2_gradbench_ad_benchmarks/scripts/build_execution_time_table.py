from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "raw_results" / "execution_time"
DETAILED_OUTPUT = PROJECT_ROOT / "raw_results" / "execution_time_measurements.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "tables" / "execution_time_table.csv"


def read_log(log_file: Path, benchmark: str, tool: str, run_name: str) -> list[dict]:
    messages: dict[int, dict] = {}
    rows: list[dict] = []
    evaluation_index = 0

    with log_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Skipping invalid JSON in {log_file}, line {line_number}: {error}")
                continue

            message = record.get("message")

            if isinstance(message, dict):
                message_id = message.get("id")

                if isinstance(message_id, int):
                    messages[message_id] = message

            response = record.get("response")

            if not isinstance(response, dict):
                continue

            response_id = response.get("id")
            original_message = messages.get(response_id)

            if not original_message:
                continue

            if original_message.get("kind") != "evaluate":
                continue

            if response.get("success") is not True:
                evaluation_index += 1
                continue

            function_name = original_message.get("function", "unknown")
            timings_by_name: dict[str, list[int]] = defaultdict(list)

            for timing in response.get("timings", []):
                timing_name = timing.get("name", "unknown")
                nanoseconds = timing.get("nanoseconds")

                if isinstance(nanoseconds, int):
                    timings_by_name[timing_name].append(nanoseconds)

            for timing_name, values in timings_by_name.items():
                median_nanoseconds = statistics.median(values)

                rows.append({
                    "benchmark": benchmark,
                    "tool": tool,
                    "run": run_name,
                    "function": function_name,
                    "timing_name": timing_name,
                    "case_index": evaluation_index,
                    "nanoseconds": median_nanoseconds,
                    "milliseconds": median_nanoseconds / 1_000_000,
                })

            evaluation_index += 1

    return rows

def collect_measurements() -> list[dict]:
    rows: list[dict] = []

    for run_directory in sorted(INPUT_ROOT.glob("run_*")):
        if not run_directory.is_dir():
            continue

        for benchmark_directory in sorted(run_directory.iterdir()):
            if not benchmark_directory.is_dir():
                continue

            for log_file in sorted(benchmark_directory.glob("*.jsonl")):
                rows.extend(
                    read_log(
                        log_file=log_file,
                        benchmark=benchmark_directory.name,
                        tool=log_file.stem,
                        run_name=run_directory.name,
                    )
                )

    return rows


def write_detailed_table(rows: list[dict]) -> None:
    DETAILED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark",
        "tool",
        "run",
        "function",
        "timing_name",
        "case_index",
        "nanoseconds",
        "milliseconds",
    ]

    with DETAILED_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_table(rows: list[dict]) -> None:
    grouped: dict[tuple, list[float]] = defaultdict(list)

    for row in rows:
        key = (
            row["benchmark"],
            row["tool"],
            row["function"],
            row["timing_name"],
            row["case_index"],
        )

        grouped[key].append(row["milliseconds"])

    summary_rows = []

    for key, values in sorted(grouped.items()):
        benchmark, tool, function_name, timing_name, case_index = key

        summary_rows.append({
            "benchmark": benchmark,
            "tool": tool,
            "function": function_name,
            "timing_name": timing_name,
            "case_index": case_index,
            "run_count": len(values),
            "mean_time_ms": statistics.mean(values),
            "median_time_ms": statistics.median(values),
            "min_time_ms": min(values),
            "max_time_ms": max(values),
            "stdev_time_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        })

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark",
        "tool",
        "function",
        "timing_name",
        "case_index",
        "run_count",
        "mean_time_ms",
        "median_time_ms",
        "min_time_ms",
        "max_time_ms",
        "stdev_time_ms",
    ]

    with SUMMARY_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    rows = collect_measurements()

    if not rows:
        raise SystemExit(f"No timing measurements found in {INPUT_ROOT}")

    write_detailed_table(rows)
    write_summary_table(rows)

    print(f"Collected measurements: {len(rows)}")
    print(f"Detailed results: {DETAILED_OUTPUT}")
    print(f"Summary table: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()