## Problem Description

The root-cause expression is:

```python
jnp.power(jnp.exp(s0), x[1])
```

Mathematically, since `exp(s0)` is positive for real `s0`, the expression can be rewritten as:

```text
(exp(s0)) ** x[1] = exp(x[1] * s0)
```

Therefore, the manual optimization is:

```python
jnp.exp(x[1] * s0)
```

## Herbie Optimization

I gave Herbie the following FPCore expression:

```lisp
(FPCore (s0 x1)
  :name "ex5 Rewriting the Power of Exponential"
  :precision binary64
  (pow (exp s0) x1))
```

Herbie returned:

```lisp
(exp (* x1 s0))
```

This is equivalent to the manual optimization:

```python
jnp.exp(x[1] * s0)
```

Therefore, the Manual and Herbie versions use the same algebraic rewrite and are expected to produce identical results.

## Input Domain

The complete function contains:

```python
jnp.arccos(x)
```

Since the function now operates on real-valued inputs, every component of `x` must satisfy:

```text
-1 <= x[i] <= 1
```

In this experiment, `x[0]` was varied. The remaining input values were selected from inside the valid domain.

Two small-number sweeps were performed:

1. An extreme near-zero stress sweep, starting from the smallest positive normal `float64` value.
2. A less extreme sweep starting from `1e-6`, where all primal and derivative results remained finite.

## Extreme Near-Zero Stress Sweep

The sweep contained 1,000 values between the smallest positive normal `float64` number and `1`.

### Primal Results

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
0

Original vs Manual NaN mismatches:
0

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
0.0

Original vs Manual max finite difference:
0.0

Manual vs Herbie max finite difference:
0.0
```

All three primal implementations produced identical finite results for all 1,000 test cases.

This indicates that the rewrite did not change the computed primal output in this range.

### First-Order Derivative Results

```text
All three completely NaN/Inf:
499

Original vs Herbie NaN mismatches:
0

Original vs Manual NaN mismatches:
0

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
1.1547005383792524

Varied value:
3.68142447055721e-24

Original vs Manual max finite difference:
1.1547005383792524

Manual vs Herbie max finite difference:
0.0
```

For 499 test cases, the first-order derivatives of all three versions were completely `NaN` or `Inf`.

Since all NaN-mismatch counts were zero, the rewrite did not prevent these failures. Whenever the Original derivative was completely invalid, the Manual and Herbie derivatives were also invalid.

For the jointly finite derivative values, the Original differed from both optimized versions by as much as:

```text
1.1547005383792524
```

The Manual and Herbie derivatives were identical.

### Second-Order Derivative Results

```text
All three completely NaN/Inf:
748

Original vs Herbie NaN mismatches:
0

Original vs Manual NaN mismatches:
0

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
4.1718496795330275e+93

Varied value:
4.00200345879629e-16

Original vs Manual max finite difference:
4.1718496795330275e+93

Manual vs Herbie max finite difference:
0.0
```

For 748 test cases, the second-order derivatives of all three implementations were completely invalid.

Again, the zero NaN-mismatch counts show that the rewrite did not fix these invalid cases. They most likely originate from other numerically sensitive operations in the complete function.

However, for the positions where both implementations produced finite derivatives, the maximum difference between the Original and optimized versions was extremely large:

```text
4.1718496795330275e+93
```

The Manual and Herbie second-order derivatives remained identical.

## Small-Number Sweep Starting From `1e-6`

A second experiment used 100 values beginning from `1e-6`. Unlike the extreme stress sweep, this experiment produced valid primal, first-order, and second-order results for every test case.

### Primal Results

```text
All three completely NaN/Inf:
0

NaN mismatches:
0

Original vs Herbie max finite difference:
0.0

Original vs Manual max finite difference:
0.0

Manual vs Herbie max finite difference:
0.0
```

All three primal versions produced identical results.

### First-Order Derivative Results

```text
All three completely NaN/Inf:
0

NaN mismatches:
0

Original vs Herbie max finite difference:
1.1547005383792524

Varied value:
5.705545567384481e-05

Original vs Manual max finite difference:
1.1547005383792524

Manual vs Herbie max finite difference:
0.0
```

All derivatives remained finite. The Manual and Herbie versions were identical, while the Original differed from both optimized versions.

### Second-Order Derivative Results

```text
All three completely NaN/Inf:
0

NaN mismatches:
0

Original vs Herbie max finite difference:
9.284550294640352e+26

Varied value:
1.5194482559544582e-06

Original vs Manual max finite difference:
9.284550294640352e+26

Manual vs Herbie max finite difference:
0.0
```

All second-order derivatives were finite, but the Original differed significantly from both optimized versions near the lower end of the tested range.

The Manual and Herbie results were again identical.

## Conclusion

The Manual and Herbie implementations use the same rewrite:

```text
(exp(s0)) ** x[1]  ->  exp(x[1] * s0)
```

As expected, they produced identical results in all experiments.

For primal evaluation, all three implementations produced identical results.

The most significant differences appeared in the first- and second-order derivatives. The Original derivative computations differed from the Manual and Herbie versions, especially for very small values of `x[0]`.

In the extreme near-zero sweep, many derivative results became completely `NaN` or `Inf`. However, these failures occurred in all three implementations, so this rewrite did not prevent them. They are likely caused by other sensitive operations in the complete function.

Starting the sweep from `1e-6` produced finite results for all test cases while still revealing substantial derivative differences between the Original and optimized computational graphs.

These experiments show that the rewrite changes the numerical behavior of the derivatives while preserving the primal result. However, a higher-precision reference would still be required to determine which derivative results are more accurate.
