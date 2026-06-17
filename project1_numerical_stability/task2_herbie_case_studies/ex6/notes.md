# Example 6 — Unstable Derivative of `acos(tanh(t))`

## Root Cause of Numerical Instability

The relevant expression pattern is:

```python
jnp.arccos(jnp.tanh(t))
```

Its derivative is:

```text
d/dt acos(tanh(t))
= -sech(t)^2 / sqrt(1 - tanh(t)^2)
```

When `tanh(t)` rounds to exactly `-1` or `1`, the denominator becomes:

```text
sqrt(1 - tanh(t)^2) = sqrt(0)
```

This can produce a numerical form equivalent to:

```text
0 / sqrt(0)
```

which results in `NaN`.

Mathematically, for real `t`, the derivative simplifies to:

```text
-sech(t)
```

and tends to `0` as `t -> ±inf`.

## Manual Rewrite

For real `t`, the following identity holds:

```text
acos(tanh(t)) = 2 * atan(exp(-t))
```

Therefore, the manually optimized expression is:

```python
2.0 * jnp.arctan(jnp.exp(-t0))
```

Only the primal subexpression was replaced. The first- and second-order derivatives were still generated automatically by JAX.

## Herbie Result

Herbie was given:

```lisp
(FPCore (t)
  :precision binary64
  (acos (tanh t)))
```

Herbie returned the same expression:

```lisp
(acos (tanh t))
```

Therefore, the Herbie implementation is identical to the Original implementation.

The three versions are:

```text
Original: acos(tanh(t0))
Manual:   2 * atan(exp(-t0))
Herbie:   acos(tanh(t0))
```

As expected, the Original and Herbie results were identical in every experiment.

---

## Experiment 1: Near-Zero Stress Sweep

The tested range for `x[0]` was approximately:

```text
1e-12 to 1e-6
```

with 1,000 test cases.

### Primal

```text
All three completely NaN/Inf: 0
NaN mismatches: 0

Original vs Herbie max finite difference: 0.0
Original vs Manual max finite difference: 0.0
Manual vs Herbie max finite difference: 0.0
```

All three versions produced identical primal results.

### First-Order Derivatives

```text
All three completely NaN/Inf: 0
NaN mismatches: 0

Original vs Herbie max finite difference: 0.0
Original vs Manual max finite difference: 0.0
Manual vs Herbie max finite difference: 0.0
```

All three versions also produced identical first-order derivatives in this range.

### Second-Order Derivatives

```text
All three completely NaN/Inf: 0
NaN mismatches: 0

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
1.7763568394002505e-15
Varied value:
1e-12

Manual vs Herbie max finite difference:
1.7763568394002505e-15
Varied value:
1e-12
```

The only observed difference was at the level of ordinary `float64` rounding.

---

## Experiment 2: Small-Number Sweep

The tested range for `x[0]` was approximately:

```text
1e-6 to 0.99
```

with 1,000 test cases.

### Primal

```text
All three completely NaN/Inf: 0
NaN mismatches: 0

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
0.7842329966896457
Varied value:
0.015036239501903484

Manual vs Herbie max finite difference:
0.7842329966896457
Varied value:
0.015036239501903484
```

The Original and Herbie versions remained identical.

The Manual version differed significantly at some inputs. This can happen when `tanh(t0)` rounds to exactly `1.0`, causing the Original expression to evaluate `acos(1.0)` as exactly `0`. The rewritten expression can still preserve a very small positive value. Since this value is later raised to the power `x`, the difference can be amplified in the complete program.

### First-Order Derivatives

```text
All three completely NaN/Inf: 0

Original vs Herbie NaN mismatches: 0
Original vs Manual NaN mismatches: 259
Herbie vs Manual NaN mismatches: 259

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
1.7201960901106759
Varied value:
0.01524547188885972

Manual vs Herbie max finite difference:
1.7201960901106759
Varied value:
0.01524547188885972
```

The Original and Herbie derivatives were identical.

The Manual rewrite changed the finite/`NaN` behavior in 259 test cases. The summary only records that their validity patterns differ; the raw CSV results should be inspected to determine which implementation is finite in each case.

### Second-Order Derivatives

```text
All three completely NaN/Inf: 0

Original vs Herbie NaN mismatches: 0
Original vs Manual NaN mismatches: 209
Herbie vs Manual NaN mismatches: 209

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
14.622257154609542
Varied value:
0.01524547188885972

Manual vs Herbie max finite difference:
14.622257154609542
Varied value:
0.01524547188885972
```

The second-order derivatives show the same pattern. The Original and Herbie versions were identical, while the Manual rewrite changed both the finite values and the `NaN` behavior.

---

## Experiment 3: Large Positive Sweep

The tested range for `x[0]` was approximately:

```text
1.0 to 50.0
```

with 1,000 test cases.

This range is large enough for `tanh(t0)` to saturate numerically, while avoiding the unrelated overflow that would eventually occur in `exp(x)` for much larger inputs.

### Primal

```text
All three completely NaN/Inf: 0
NaN mismatches: 0

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
1.4210854715202004e-14
Varied value:
2.3760097539893064

Manual vs Herbie max finite difference:
1.4210854715202004e-14
Varied value:
2.3760097539893064
```

The primal difference was very small and consistent with `float64` rounding.

### First-Order Derivatives

```text
All three completely NaN/Inf: 0

Original vs Herbie NaN mismatches: 0
Original vs Manual NaN mismatches: 235
Herbie vs Manual NaN mismatches: 235

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
1.4210854715202004e-14
Varied value:
2.2229864612374914

Manual vs Herbie max finite difference:
1.4210854715202004e-14
Varied value:
2.2229864612374914
```

The Manual rewrite changed the derivative validity pattern in 235 test cases, while the Original and Herbie versions remained identical.

### Second-Order Derivatives

```text
All three completely NaN/Inf: 0

Original vs Herbie NaN mismatches: 0
Original vs Manual NaN mismatches: 235
Herbie vs Manual NaN mismatches: 235

Original vs Herbie max finite difference: 0.0

Original vs Manual max finite difference:
2.842170943040401e-14
Varied value:
1.4281079323479964

Manual vs Herbie max finite difference:
2.842170943040401e-14
Varied value:
1.4281079323479964
```

The second-order results follow the same pattern as the first-order results.

---

## Conclusion

Herbie did not find a rewrite for `acos(tanh(t))`, so the Herbie and Original implementations were identical in every experiment.

The manual rewrite:

```text
acos(tanh(t)) -> 2 * atan(exp(-t))
```

preserved the primal result exactly in the near-zero sweep and very closely in the large-positive sweep, but produced a noticeable primal difference in part of the small-number sweep. It also changed the numerical behavior of the automatically generated derivatives.

The near-zero stress sweep showed almost no difference between the implementations. In the wider small-number and large-positive sweeps, the Manual version produced different derivative validity patterns from the Original and Herbie versions.

The strongest evidence is:

```text
Small-number sweep:
First-order NaN mismatches:  259
Second-order NaN mismatches: 209

Large-positive sweep:
First-order NaN mismatches:  235
Second-order NaN mismatches: 235
```

These results show that the manual rewrite changes the numerical stability of the derivative computation, while Herbie does not improve the original expression.

However, the mismatch counters do not indicate which version is finite in each individual case. The raw CSV files should be inspected before making a stronger claim that one implementation always avoids the `NaN` values. A higher-precision reference would also be required to determine which finite results are more accurate.