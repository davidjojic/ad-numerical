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
    [
        1.3 + 0.4j,
        1231.6 + 0.2j,
        0.9 + 0.5j,
        1.1 + 0.3j,
        1.4 + 0.1j,
    ],
    dtype=jnp.complex128,
)

x_real = jnp.concatenate([
    jnp.real(x),
    jnp.imag(x),
])


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

orig_real = make_real(f_original)
man_real = make_real(f_manual)
herb_real = make_real(f_herbie)

first_orig = jax.jacfwd(orig_real)(x_real)
first_man = jax.jacfwd(man_real)(x_real)
first_herb = jax.jacfwd(herb_real)(x_real)

first_man_orig = max_abs_diff(first_man, first_orig)
first_herb_orig = max_abs_diff(first_herb, first_orig)
first_man_herb = max_abs_diff(first_man, first_herb)

print("\nManual vs Original:")
print(first_man_orig)

print("\nHerbie vs Original:")
print(first_herb_orig)

print("\nManual vs Herbie:")
print(first_man_herb)


# ============================================================
# Second-order derivatives
# ============================================================

second_orig = jax.jacfwd(jax.jacfwd(orig_real))(x_real)
second_man = jax.jacfwd(jax.jacfwd(man_real))(x_real)
second_herb = jax.jacfwd(jax.jacfwd(herb_real))(x_real)

second_man_orig = max_abs_diff(second_man, second_orig)
second_herb_orig = max_abs_diff(second_herb, second_orig)
second_man_herb = max_abs_diff(second_man, second_herb)

print("\nSECOND-ORDER DERIVATIVE:")
print("Shape:", second_orig.shape)

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

input_real = to_json(jnp.real(x))
input_imag = to_json(jnp.imag(x))

primal_fields = [
    "run_id",
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

primal_row = {
    "run_id": run_id,
    "input_real": input_real,
    "input_imag": input_imag,

    "original_output_real": to_json(jnp.real(f_orig)),
    "original_output_imag": to_json(jnp.imag(f_orig)),

    "manual_output_real": to_json(jnp.real(f_man)),
    "manual_output_imag": to_json(jnp.imag(f_man)),

    "herbie_output_real": to_json(jnp.real(f_herb)),
    "herbie_output_imag": to_json(jnp.imag(f_herb)),

    "manual_vs_original": primal_man_orig,
    "herbie_vs_original": primal_herb_orig,
    "manual_vs_herbie": primal_man_herb,
}

append_csv(
    results_dir / "primal_results.csv",
    primal_fields,
    primal_row,
)

first_fields = [
    "run_id",
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

first_row = {
    "run_id": run_id,
    "input_real": input_real,
    "input_imag": input_imag,
    "shape": str(first_orig.shape),

    "original_derivative": to_json(first_orig),
    "manual_derivative": to_json(first_man),
    "herbie_derivative": to_json(first_herb),

    "manual_vs_original": first_man_orig,
    "herbie_vs_original": first_herb_orig,
    "manual_vs_herbie": first_man_herb,
}

append_csv(
    results_dir / "first_order_derivative_results.csv",
    first_fields,
    first_row,
)


second_fields = [
    "run_id",
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

second_row = {
    "run_id": run_id,
    "input_real": input_real,
    "input_imag": input_imag,
    "shape": str(second_orig.shape),

    "original_derivative": to_json(second_orig),
    "manual_derivative": to_json(second_man),
    "herbie_derivative": to_json(second_herb),

    "manual_vs_original": second_man_orig,
    "herbie_vs_original": second_herb_orig,
    "manual_vs_herbie": second_man_herb,
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
