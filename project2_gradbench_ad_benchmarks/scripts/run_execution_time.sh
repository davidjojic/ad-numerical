#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRADBENCH_DIR="$PROJECT_ROOT/gradbench"
OUTPUT_ROOT="$PROJECT_ROOT/raw_results/execution_time"

EVALS=("gmm")
TOOLS=("manual" "jax" "pytorch")
NUMBER_OF_RUNS=3

mkdir -p "$OUTPUT_ROOT"
cd "$GRADBENCH_DIR"

for run_number in $(seq 1 "$NUMBER_OF_RUNS"); do
    output_directory="$OUTPUT_ROOT/run_$run_number"

    echo "========================================"
    echo "Starting run $run_number/$NUMBER_OF_RUNS"
    echo "Output: $output_directory"
    echo "========================================"

    rm -rf "$output_directory"

    arguments=(
        repo run
        --check
        --output "$output_directory"
    )

    for eval_name in "${EVALS[@]}"; do
        arguments+=(--eval "$eval_name")
    done

    for tool_name in "${TOOLS[@]}"; do
        arguments+=(--tool "$tool_name")
    done

    ./gradbench "${arguments[@]}"
done

echo "========================================"
echo "All execution-time runs completed."
echo "Results: $OUTPUT_ROOT"
echo "========================================"