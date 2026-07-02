#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRADBENCH_DIR="$PROJECT_ROOT/gradbench"
COMBINATIONS_FILE="$PROJECT_ROOT/tables/selected_combinations.csv"
OUTPUT_ROOT="$PROJECT_ROOT/raw_results/execution_time"
STATUS_FILE="$PROJECT_ROOT/raw_results/execution_time_status.csv"

NUMBER_OF_RUNS=3

echo "benchmark,tool,run,status" > "$STATUS_FILE"

while IFS=, read -r benchmark tool; do
    if [[ "$benchmark" == "benchmark" ]]; then
        continue
    fi

    for run_number in $(seq 1 "$NUMBER_OF_RUNS"); do
        output_directory="$OUTPUT_ROOT/run_$run_number"
        expected_log="$output_directory/$benchmark/$tool.jsonl"

        if [[ -s "$expected_log" ]]; then
            echo "Skipping existing result: $benchmark / $tool / run $run_number"
            echo "$benchmark,$tool,$run_number,existing" >> "$STATUS_FILE"
            continue
        fi

        echo "=================================================="
        echo "Benchmark: $benchmark"
        echo "Tool:      $tool"
        echo "Run:       $run_number/$NUMBER_OF_RUNS"
        echo "=================================================="

        cd "$GRADBENCH_DIR"

        if ./gradbench repo run \
            --eval "$benchmark" \
            --tool "$tool" \
            --output "$output_directory" \
            --check; then

            echo "$benchmark,$tool,$run_number,success" >> "$STATUS_FILE"
        else
            echo "$benchmark,$tool,$run_number,failed" >> "$STATUS_FILE"
            echo "Combination failed; continuing with the next one."
        fi
    done
done < "$COMBINATIONS_FILE"

echo "Finished."
echo "Status saved to: $STATUS_FILE"