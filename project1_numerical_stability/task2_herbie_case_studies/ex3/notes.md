## Problem Statement

The original function contains the following expression:

```python
s0 = sqrt(x2)
t0[2] = (s0 / x2) * s0 + x2
```

## Mathematical Optimization

For `x2 > 0`, we can simplify the expression as follows:

```text
(sqrt(x2) / x2) * sqrt(x2) + x2
= (sqrt(x2) * sqrt(x2)) / x2 + x2
= x2 / x2 + x2
= 1 + x2
```

Therefore, the manual optimization is:

```python
t0[2] = 1 + x2
```

## Herbie Optimization

I gave Herbie the following expression:

```lisp
:pre (> x 0.0)
(let ([s0 (sqrt x)])
  (+ (* (/ s0 x) s0) x))
```

Herbie returned:

```lisp
:pre (> x 0.0)
(+ x 1.0)
```

This is equivalent to:

```python
t0[2] = x2 + 1
```

When compared with the manual optimization, the only difference is the order of addition. Since the two expressions are identical in floating-point arithmetic for these operands, we should not expect any difference between the Manual and Herbie results.

## Primal Results

### Small values of `x[2]`

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
2.0
Varied value:
2.2250738585072014e-308

Original vs Manual max finite difference:
2.0
Varied value:
2.2250738585072014e-308

Manual vs Herbie max finite difference:
0.0
Varied value:
2.2250738585072014e-308
```

The Manual and Herbie versions produced identical results. The maximum finite difference between the original and optimized versions occurred at the smallest tested positive `float64` value.

### Large values of `x[2]`

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
14142135622.316736
Varied value:
10000000000.0

Original vs Manual max finite difference:
14142135622.316736
Varied value:
10000000000.0

Manual vs Herbie max finite difference:
0.0
Varied value:
1.0
```

The Manual and Herbie versions again produced identical results. As `x[2]` became larger, the absolute difference between the original and optimized versions also increased.

However, these results compare the output of the complete program rather than only the simplified subexpression. Therefore, the large difference may be caused by a small rounding difference being amplified by later operations in the program.

Also, absolute difference alone is not enough to determine which result is more accurate. A relative-error comparison or a higher-precision reference would be needed.

## First-Order Derivative Results

### Small values of `x[2]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
500

Original vs Manual NaN mismatches:
500

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
7.442828536787015e+137
Varied value:
2.1264370166982005e-154

Original vs Manual max finite difference:
7.442828536787015e+137
Varied value:
2.1264370166982005e-154

Manual vs Herbie max finite difference:
0.0
Varied value:
2.2250738585072014e-308
```

The Manual and Herbie derivatives were identical. However, the original function had 500 NaN mismatches compared with both optimized versions.

This means that for approximately half of the tested small values, the original derivative contained at least one `NaN` or `Inf` value where the optimized versions produced a finite value.

The maximum finite difference between the original and optimized derivatives was also extremely large, suggesting that the original computational form becomes unstable when `x[2]` is very close to zero.

### Large values of `x[2]`

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
10000000000.0
Varied value:
10000000000.0

Original vs Manual max finite difference:
10000000000.0
Varied value:
10000000000.0

Manual vs Herbie max finite difference:
0.0
Varied value:
1.0
```

For large values, all three versions remained finite. The absolute difference between the original and optimized derivatives increased as `x[2]` increased, while the Manual and Herbie derivatives remained identical.

## Second-Order Derivative Results

### Small values of `x[2]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
667

Original vs Manual NaN mismatches:
667

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
4.455508415646675e+189
Varied value:
1.915615703511538e-206

Original vs Manual max finite difference:
4.455508415646675e+189
Varied value:
1.915615703511538e-206

Manual vs Herbie max finite difference:
0.0
Varied value:
2.2250738585072014e-308
```

The second-order derivatives show even stronger instability near zero. The original version had 667 NaN mismatches compared with both optimized versions, while the Manual and Herbie results remained identical.

The maximum finite difference between the original and optimized versions was also extremely large.

### Large values of `x[2]`

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
10000000000.0
Varied value:
10000000000.0

Original vs Manual max finite difference:
10000000000.0
Varied value:
10000000000.0

Manual vs Herbie max finite difference:
0.0
Varied value:
1.0
```

The behavior for large values is similar to the first-order results. All versions remained finite, but the absolute difference between the original and optimized versions increased with `x[2]`.

## Conclusion

The Manual and Herbie optimizations are mathematically equivalent for `x[2] > 0`, and they produced identical results in every experiment.

The most important improvement appears for very small values of `x[2]`. Although no test case caused all three versions to become completely `NaN` or `Inf`, the original version produced many NaN mismatches:

```text
First-order derivatives: 500 mismatches
Second-order derivatives: 667 mismatches
```

This indicates that the original computational form becomes numerically unstable near zero, especially when computing derivatives, while the Manual and Herbie versions continue to produce finite results.

For large values of `x[2]`, no NaN mismatches occurred. However, the absolute differences between the original and optimized versions increased as `x[2]` became larger. Since the results are taken from the complete program, these differences may be caused by rounding errors being amplified by later calculations.

Overall, the rewrite provides strong evidence of improved numerical stability near zero. A relative-error calculation or comparison against a higher-precision reference would still be required to determine which implementation is more accurate.
