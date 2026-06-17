import csv
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_fn_original import fn as f_original
from jax_fn_herbie import fn as f_herbie
from jax_fn_manual import fn as f_manual


jax.config.update("jax_enable_x64", True)


# ============================================================
# Helpers
# ============================================================

def make_real(f):
    def wrapped(z):
        x_complex = z[:5] + 1j * z[5:]
        y_complex = f(x_complex)

        return jnp.concatenate([
            jnp.real(y_complex),
            jnp.imag(y_complex),
        ])

    return wrapped


def to_json(array):
    host_array = jax.device_get(array)
    return json.dumps(host_array.tolist())


def append_csv(file_path, fieldnames, row):
    file_exists = file_path.exists() and file_path.stat().st_size > 0

    with file_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


"""
If this array has a finite resolt
"""
def has_finite_value(array):
    return bool(jnp.any(jnp.isfinite(array)))


def all_results_are_invalid(*arrays):
    """
    True if all results are NaN/Inf
    """
    return all(
        not has_finite_value(array)
        for array in arrays
    )


def has_nan_mismatch(first, second):
    """
        If one has a NaN but other has a finite result
    """
    first_finite = jnp.isfinite(first)
    second_finite = jnp.isfinite(second)

    return bool(jnp.any(first_finite != second_finite))


def max_finite_abs_diff(first, second):
    """
    The biggest abs diff
    """
    finite_positions = (
        jnp.isfinite(first)
        & jnp.isfinite(second)
    )

    if not bool(jnp.any(finite_positions)):
        return None

    differences = jnp.abs(first - second)

    valid_differences = jnp.where(
        finite_positions,
        differences,
        -jnp.inf,
    )

    return float(jnp.max(valid_differences))


def create_statistics():
    return {
        "all_nan": 0,

        "org_vs_herbie_nan": 0,
        "org_vs_man_nan": 0,
        "herbie_vs_man_nan": 0,

        "org_vs_herbie_max_diff": None,
        "org_vs_man_max_diff": None,
        "man_vs_herbie_max_diff": None,

        "org_vs_herbie_max_diff_value": None,
        "org_vs_man_max_diff_value": None,
        "man_vs_herbie_max_diff_value": None,
    }


def update_maximum(
    statistics,
    diff_name,
    value_name,
    difference,
    value,
):
    if difference is None:
        return

    old_difference = statistics[diff_name]

    if old_difference is None or difference > old_difference:
        statistics[diff_name] = difference
        statistics[value_name] = float(value)


def update_statistics(
    statistics,
    original,
    manual,
    herbie,
    value,
):
    # If all the funciton returns only NaN
    if all_results_are_invalid(
        original,
        manual,
        herbie,
    ):
        statistics["all_nan"] += 1

    # We count when one version is finite but other has NaN
    if has_nan_mismatch(original, herbie):
        statistics["org_vs_herbie_nan"] += 1

    if has_nan_mismatch(original, manual):
        statistics["org_vs_man_nan"] += 1

    if has_nan_mismatch(herbie, manual):
        statistics["herbie_vs_man_nan"] += 1

    org_herbie_diff = max_finite_abs_diff(
        original,
        herbie,
    )

    org_man_diff = max_finite_abs_diff(
        original,
        manual,
    )

    man_herbie_diff = max_finite_abs_diff(
        manual,
        herbie,
    )

    update_maximum(
        statistics,
        "org_vs_herbie_max_diff",
        "org_vs_herbie_max_diff_value",
        org_herbie_diff,
        value,
    )

    update_maximum(
        statistics,
        "org_vs_man_max_diff",
        "org_vs_man_max_diff_value",
        org_man_diff,
        value,
    )

    update_maximum(
        statistics,
        "man_vs_herbie_max_diff",
        "man_vs_herbie_max_diff_value",
        man_herbie_diff,
        value,
    )


# ============================================================
# Generate many x values
# ============================================================

NUMBER_OF_TESTS = 1000

# What index are we cheking
VARIED_INDEX = 1


# What test we are generating
# small_num_sweep
# large_num_sweep
SWEEP_NAME = "large_num_sweep"
INCLUDE_ZERO = False # will we include 0 in our test cases
START_VALUE = 1.0
END_VALUE = 1e10


base_x = jnp.array(
    [
        1.0 + 1j,
        1.0 + 1.0j,
        1.0 + 1.0j,
        1.0 + 1.0j,
        0.0 + 0.0j,
    ],
    dtype=jnp.complex128,
)


# geomspace we use to callcuclate the x[index]
positive_values = jnp.geomspace(
    START_VALUE,
    END_VALUE,
    NUMBER_OF_TESTS - 1,
    dtype=jnp.float64,
)


if INCLUDE_ZERO:
    positive_values = jnp.geomspace(
        START_VALUE,
        END_VALUE,
        NUMBER_OF_TESTS - 1,
        dtype=jnp.float64,
    )

    test_values = jnp.concatenate([
        jnp.array([0.0], dtype=jnp.float64),
        positive_values,
    ])
else:
    test_values = jnp.geomspace(
        START_VALUE,
        END_VALUE,
        NUMBER_OF_TESTS,
        dtype=jnp.float64,
    )


# ============================================================
# Prepare functions
# ============================================================

orig_real = make_real(f_original)
man_real = make_real(f_manual)
herb_real = make_real(f_herbie)


original_fn = jax.jit(f_original)
manual_fn = jax.jit(f_manual)
herbie_fn = jax.jit(f_herbie)

first_original_fn = jax.jit(
    jax.jacfwd(orig_real)
)

first_manual_fn = jax.jit(
    jax.jacfwd(man_real)
)

first_herbie_fn = jax.jit(
    jax.jacfwd(herb_real)
)

second_original_fn = jax.jit(
    jax.jacfwd(
        jax.jacfwd(orig_real)
    )
)

second_manual_fn = jax.jit(
    jax.jacfwd(
        jax.jacfwd(man_real)
    )
)

second_herbie_fn = jax.jit(
    jax.jacfwd(
        jax.jacfwd(herb_real)
    )
)


# ============================================================
# Prepare CSV files
# ============================================================

results_dir = Path("results")
results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

batch_id = datetime.now().isoformat(
    timespec="seconds"
)

primal_path = (
    results_dir
    / f"{SWEEP_NAME}_primal_results.csv"
)

first_order_path = (
    results_dir
    / f"{SWEEP_NAME}_first_order_results.csv"
)

second_order_path = (
    results_dir
    / f"{SWEEP_NAME}_second_order_results.csv"
)

summary_path = (
    results_dir
    / f"{SWEEP_NAME}_summary.csv"
)


primal_fields = [
    "batch_id",
    "test_id",
    "varied_index",
    "varied_value",
    "input_real",
    "input_imag",
    "original_output_real",
    "original_output_imag",
    "manual_output_real",
    "manual_output_imag",
    "herbie_output_real",
    "herbie_output_imag",
    "manual_vs_original",
    "herbie_vs_original",
    "manual_vs_herbie",
]


first_fields = [
    "batch_id",
    "test_id",
    "varied_index",
    "varied_value",
    "input_real",
    "input_imag",
    "shape",
    "original_derivative",
    "manual_derivative",
    "herbie_derivative",
    "manual_vs_original",
    "herbie_vs_original",
    "manual_vs_herbie",
]


second_fields = [
    "batch_id",
    "test_id",
    "varied_index",
    "varied_value",
    "input_real",
    "input_imag",
    "shape",
    "original_derivative",
    "manual_derivative",
    "herbie_derivative",
    "manual_vs_original",
    "herbie_vs_original",
    "manual_vs_herbie",
]


summary_fields = [
    "batch_id",
    "result_type",
    "all_nan",
    "org_vs_herbie_nan",
    "org_vs_man_nan",
    "herbie_vs_man_nan",
    "org_vs_herbie_max_diff",
    "org_vs_man_max_diff",
    "man_vs_herbie_max_diff",
    "org_vs_herbie_max_diff_value",
    "org_vs_man_max_diff_value",
    "man_vs_herbie_max_diff_value",
]


# ============================================================
# Statistics
# ============================================================

statistics = {
    "primal": create_statistics(),
    "first_order": create_statistics(),
    "second_order": create_statistics(),
}

saved_tests = 0
skipped_tests = 0
failed_tests = 0


# ============================================================
# Run all tests
# ============================================================

for test_id, value in enumerate(test_values):

    #We keep img part of input, but change only the real part
    old_imaginary_part = jnp.imag(
        base_x[VARIED_INDEX]
    )

    current_x = base_x.at[VARIED_INDEX].set(
        value + 1j * old_imaginary_part
    )

    current_x_real = jnp.concatenate([
        jnp.real(current_x),
        jnp.imag(current_x),
    ])

    try:
        # ====================================================
        # Primal
        # ====================================================

        f_orig = original_fn(current_x)
        f_man = manual_fn(current_x)
        f_herb = herbie_fn(current_x)

        # ====================================================
        # First-order derivatives
        # ====================================================

        first_orig = first_original_fn(
            current_x_real
        )

        first_man = first_manual_fn(
            current_x_real
        )

        first_herb = first_herbie_fn(
            current_x_real
        )

        # ====================================================
        # Second-order derivatives
        # ====================================================

        second_orig = second_original_fn(
            current_x_real
        )

        second_man = second_manual_fn(
            current_x_real
        )

        second_herb = second_herbie_fn(
            current_x_real
        )

        # Sačekaj da JAX završi računanje.
        second_orig.block_until_ready()
        second_man.block_until_ready()
        second_herb.block_until_ready()

    except Exception as error:
        failed_tests += 1

        print(
            f"Test {test_id} failed "
            f"for value {float(value)}:"
        )

        print(error)
        continue

    # ========================================================
    # Update statistics
    # ========================================================

    update_statistics(
        statistics["primal"],
        f_orig,
        f_man,
        f_herb,
        value,
    )

    update_statistics(
        statistics["first_order"],
        first_orig,
        first_man,
        first_herb,
        value,
    )

    update_statistics(
        statistics["second_order"],
        second_orig,
        second_man,
        second_herb,
        value,
    )
    if all_results_are_invalid(
        f_orig,
        f_man,
        f_herb,
        first_orig,
        first_man,
        first_herb,
        second_orig,
        second_man,
        second_herb,
    ):
        skipped_tests += 1
        continue

    # ========================================================
    # Calculate finite differences
    # ========================================================

    primal_man_orig = max_finite_abs_diff(
        f_man,
        f_orig,
    )

    primal_herb_orig = max_finite_abs_diff(
        f_herb,
        f_orig,
    )

    primal_man_herb = max_finite_abs_diff(
        f_man,
        f_herb,
    )


    first_man_orig = max_finite_abs_diff(
        first_man,
        first_orig,
    )

    first_herb_orig = max_finite_abs_diff(
        first_herb,
        first_orig,
    )

    first_man_herb = max_finite_abs_diff(
        first_man,
        first_herb,
    )


    second_man_orig = max_finite_abs_diff(
        second_man,
        second_orig,
    )

    second_herb_orig = max_finite_abs_diff(
        second_herb,
        second_orig,
    )

    second_man_herb = max_finite_abs_diff(
        second_man,
        second_herb,
    )

    # ========================================================
    # Common input information
    # ========================================================

    input_real = to_json(
        jnp.real(current_x)
    )

    input_imag = to_json(
        jnp.imag(current_x)
    )

    # ========================================================
    # Save primal
    # ========================================================

    primal_row = {
        "batch_id": batch_id,
        "test_id": test_id,
        "varied_index": VARIED_INDEX,
        "varied_value": float(value),
        "input_real": input_real,
        "input_imag": input_imag,

        "original_output_real": to_json(
            jnp.real(f_orig)
        ),

        "original_output_imag": to_json(
            jnp.imag(f_orig)
        ),

        "manual_output_real": to_json(
            jnp.real(f_man)
        ),

        "manual_output_imag": to_json(
            jnp.imag(f_man)
        ),

        "herbie_output_real": to_json(
            jnp.real(f_herb)
        ),

        "herbie_output_imag": to_json(
            jnp.imag(f_herb)
        ),

        "manual_vs_original": primal_man_orig,
        "herbie_vs_original": primal_herb_orig,
        "manual_vs_herbie": primal_man_herb,
    }

    append_csv(
        primal_path,
        primal_fields,
        primal_row,
    )

    # ========================================================
    # Save first-order derivative
    # ========================================================

    first_row = {
        "batch_id": batch_id,
        "test_id": test_id,
        "varied_index": VARIED_INDEX,
        "varied_value": float(value),
        "input_real": input_real,
        "input_imag": input_imag,
        "shape": str(first_orig.shape),

        "original_derivative": to_json(
            first_orig
        ),

        "manual_derivative": to_json(
            first_man
        ),

        "herbie_derivative": to_json(
            first_herb
        ),

        "manual_vs_original": first_man_orig,
        "herbie_vs_original": first_herb_orig,
        "manual_vs_herbie": first_man_herb,
    }

    append_csv(
        first_order_path,
        first_fields,
        first_row,
    )

    # ========================================================
    # Save second-order derivative
    # ========================================================

    second_row = {
        "batch_id": batch_id,
        "test_id": test_id,
        "varied_index": VARIED_INDEX,
        "varied_value": float(value),
        "input_real": input_real,
        "input_imag": input_imag,
        "shape": str(second_orig.shape),

        "original_derivative": to_json(
            second_orig
        ),

        "manual_derivative": to_json(
            second_man
        ),

        "herbie_derivative": to_json(
            second_herb
        ),

        "manual_vs_original": second_man_orig,
        "herbie_vs_original": second_herb_orig,
        "manual_vs_herbie": second_man_herb,
    }

    append_csv(
        second_order_path,
        second_fields,
        second_row,
    )

    saved_tests += 1

    if test_id % 100 == 0:
        print(
            f"Completed "
            f"{test_id}/{NUMBER_OF_TESTS} tests"
        )


# ============================================================
# Save summary CSV
# ============================================================

for result_type, values in statistics.items():

    summary_row = {
        "batch_id": batch_id,
        "result_type": result_type,

        "all_nan": values["all_nan"],

        "org_vs_herbie_nan":
            values["org_vs_herbie_nan"],

        "org_vs_man_nan":
            values["org_vs_man_nan"],

        "herbie_vs_man_nan":
            values["herbie_vs_man_nan"],

        "org_vs_herbie_max_diff":
            values["org_vs_herbie_max_diff"],

        "org_vs_man_max_diff":
            values["org_vs_man_max_diff"],

        "man_vs_herbie_max_diff":
            values["man_vs_herbie_max_diff"],

        "org_vs_herbie_max_diff_value":
            values["org_vs_herbie_max_diff_value"],

        "org_vs_man_max_diff_value":
            values["org_vs_man_max_diff_value"],

        "man_vs_herbie_max_diff_value":
            values["man_vs_herbie_max_diff_value"],
    }

    append_csv(
        summary_path,
        summary_fields,
        summary_row,
    )


# ============================================================
# Final terminal output
# ============================================================

print("\n========================================")
print("FINISHED")
print("========================================")

print(
    "Total generated tests:",
    NUMBER_OF_TESTS,
)

print(
    "Saved tests:",
    saved_tests,
)

print(
    "Skipped completely invalid tests:",
    skipped_tests,
)

print(
    "Tests that raised an exception:",
    failed_tests,
)


for result_type, values in statistics.items():

    print("\n----------------------------------------")
    print(result_type.upper())
    print("----------------------------------------")

    print("All three completely NaN/Inf:")
    print(values["all_nan"])

    print("\nOriginal vs Herbie NaN mismatches:")
    print(values["org_vs_herbie_nan"])

    print("\nOriginal vs Manual NaN mismatches:")
    print(values["org_vs_man_nan"])

    print("\nHerbie vs Manual NaN mismatches:")
    print(values["herbie_vs_man_nan"])

    print(
        "\nOriginal vs Herbie "
        "max finite difference:"
    )

    print(
        values["org_vs_herbie_max_diff"]
    )

    print("Varied value:")

    print(
        values[
            "org_vs_herbie_max_diff_value"
        ]
    )

    print(
        "\nOriginal vs Manual "
        "max finite difference:"
    )

    print(
        values["org_vs_man_max_diff"]
    )

    print("Varied value:")

    print(
        values[
            "org_vs_man_max_diff_value"
        ]
    )

    print(
        "\nManual vs Herbie "
        "max finite difference:"
    )

    print(
        values["man_vs_herbie_max_diff"]
    )

    print("Varied value:")

    print(
        values[
            "man_vs_herbie_max_diff_value"
        ]
    )


print("\nResults saved in:")

print(primal_path)

print(first_order_path)

print(second_order_path)

print(summary_path)