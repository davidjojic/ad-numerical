import csv
import json
import urllib.request
from pathlib import Path


SUMMARY_URL = "https://raw.githubusercontent.com/gradbench/gradbench/refs/heads/ci/refs/heads/nightly/summary.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "tables" / "supported_combinations.csv"


def main() -> None:
    with urllib.request.urlopen(SUMMARY_URL) as response:
        summary = json.load(response)

    rows = []

    for eval_result in summary["table"]:
        benchmark = eval_result["eval"]

        if benchmark == "hello":
            continue

        for tool_result in eval_result["tools"]:
            if "score" not in tool_result:
                continue

            rows.append({
                "benchmark": benchmark,
                "tool": tool_result["tool"],
                "nightly_score": tool_result["score"],
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["benchmark", "tool", "nightly_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Nightly date: {summary['date']}")
    print(f"Nightly commit: {summary['commit']}")
    print(f"Supported combinations: {len(rows)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()