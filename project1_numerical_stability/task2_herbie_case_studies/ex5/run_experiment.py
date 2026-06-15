import csv
import json
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax_fn_original import fn as f_original
from jax_fn_herbie import fn as f_herbie
from jax_fn_manual import fn as f_manual


def to_json(array):
    
    host_array = jax.device_get(array)
    return json.dumps(host_array.tolist())


def max_abs_diff(first, second):
    
    return float(jnp.max(jnp.abs(first - second)))


def append_csv(file_path, fieldnames, row):
    
    file_exists = file_path.exists() and file_path.stat().st_size > 0

    with file_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Input
# ============================================================

x = jnp.array(
    [9.0, 0.000005, 0.2, 0.3, 0.4],
    dtype=jnp.float64,
)

# ============================================================
# Primal outputs
# ============================================================

f_orig = f_original(x)
f_man = f_manual(x)
f_herb = f_herbie(x)

primal_man_orig = max_abs_diff(f_man, f_orig)
primal_herb_orig = max_abs_diff(f_herb, f_orig)
primal_man_herb = max_abs_diff(f_man, f_herb)

print("\nPRIMAL:")

print("\nManual vs Original:")
print(primal_man_orig)

print("\nHerbie vs Original:")
print(primal_herb_orig)

print("\nManual vs Herbie:")
print(primal_man_herb)


# ============================================================
# First-order derivatives

# ============================================================

first_orig = jax.jacfwd(f_original)(x)
first_man = jax.jacfwd(f_manual)(x)
first_herb = jax.jacfwd(f_herbie)(x)

first_man_orig = max_abs_diff(first_man, first_orig)
first_herb_orig = max_abs_diff(first_herb, first_orig)
first_man_herb = max_abs_diff(first_man, first_herb)

print("\nFIRST-ORDER DERIVATIVE:")

print("\nManual vs Original:")
print(first_man_orig)

print("\nHerbie vs Original:")
print(first_herb_orig)

print("\nManual vs Herbie:")
print(first_man_herb)


# ============================================================
# Second-order derivatives
# ============================================================

second_orig = jax.jacfwd(jax.jacfwd(f_original))(x)
second_man = jax.jacfwd(jax.jacfwd(f_manual))(x)
second_herb = jax.jacfwd(jax.jacfwd(f_herbie))(x)

second_man_orig = max_abs_diff(second_man, second_orig)
second_herb_orig = max_abs_diff(second_herb, second_orig)
second_man_herb = max_abs_diff(second_man, second_herb)


print("\nManual vs Original:")
print(second_man_orig)

print("\nHerbie vs Original:")
print(second_herb_orig)

print("\nManual vs Herbie:")
print(second_man_herb)


# ============================================================
# Save results
# ============================================================

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

run_id = datetime.now().isoformat(timespec="seconds")
input_values = to_json(x)


primal_fields = [
    "run_id",
    "input",
    "original_output",
    "manual_output",
    "herbie_output",
]

primal_row = {
    "run_id": run_id,
    "input": input_values,
    "original_output": to_json(f_orig),
    "manual_output": to_json(f_man),
    "herbie_output": to_json(f_herb),
}

append_csv(
    results_dir / "primal_results.csv",
    primal_fields,
    primal_row,
)

first_fields = [
    "run_id",
    "input",
    "shape",
    "original_derivative",
    "manual_derivative",
    "herbie_derivative",
]

first_row = {
    "run_id": run_id,
    "input": input_values,
    "shape": str(first_orig.shape),
    "original_derivative": to_json(first_orig),
    "manual_derivative": to_json(first_man),
    "herbie_derivative": to_json(first_herb),
}

append_csv(
    results_dir / "first_order_derivative_results.csv",
    first_fields,
    first_row,
)

second_fields = [
    "run_id",
    "input",
    "shape",
    "original_derivative",
    "manual_derivative",
    "herbie_derivative",
]

second_row = {
    "run_id": run_id,
    "input": input_values,
    "shape": str(second_orig.shape),
    "original_derivative": to_json(second_orig),
    "manual_derivative": to_json(second_man),
    "herbie_derivative": to_json(second_herb),
}

append_csv(
    results_dir / "second_order_derivative_results.csv",
    second_fields,
    second_row,
)


print("\nResults saved in:")
print(results_dir / "primal_results.csv")
print(results_dir / "first_order_derivative_results.csv")
print(results_dir / "second_order_derivative_results.csv")