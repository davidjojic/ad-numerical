## Root Cause of Numerical Instability

The relevant expressions are:

```python
s1 = x[1] ** x[2]
s2 = x[1] ** (x[1] * x[1] - cos(x[1]))

t2 = s1 + t1
t3 = s2 + t2

out = x[2] ** (x[2] + t1) - s2 + t3
```

Since:

```python
t3 = s2 + s1 + t1
```

we can substitute `t3` into the output expression:

```text
out = x[2] ** (x[2] + t1) - s2 + (s2 + s1 + t1)
```

The `-s2 + s2` terms cancel, so the expression becomes:

```text
out = x[2] ** (x[2] + t1) + s1 + t1
```

Since:

```python
s1 = x[1] ** x[2]
```

the manual optimization is:

```python
out = x[2] ** (x[2] + t1) + x[1] ** x[2] + t1
```

## Herbie Optimization

I gave Herbie the following expression:

```lisp
(let* ([s1 (pow x1 x2)]
       [s2 (pow x1 (- (* x1 x1) (cos x1)))]
       [t2 (+ s1 t1)]
       [t3 (+ s2 t2)])
  (+ (- (pow x2 (+ x2 t1)) s2) t3))
```

Herbie returned:

```lisp
(+ t1
   (+ (pow x2 (+ t1 x2))
      (pow x1 x2)))
```

In Python, this is:

```python
out = t1 + (
    x[2] ** (t1 + x[2])
    + x[1] ** x[2]
)
```

This is mathematically equivalent to the manual optimization. The only difference is the order of the additions.

## Herbie vs. Manual Optimization

The Manual and Herbie versions produced identical results in all of the experiments:

```text
Manual vs Herbie max finite difference: 0.0
Manual vs Herbie NaN mismatches: 0
```

This means that the different order of addition did not introduce an observable numerical difference in these test cases.

## Primal Results

### Small values of `x[1]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
1

Original vs Manual NaN mismatches:
1

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
4.6603507209893245
Varied value:
2.2250738585072014e-308

Original vs Manual max finite difference:
4.6603507209893245
Varied value:
2.2250738585072014e-308

Manual vs Herbie max finite difference:
0.0
```

The Manual and Herbie versions produced identical primal results.

The maximum finite difference between the original and optimized versions occurred at the smallest tested positive `float64` value.

There was also one NaN mismatch, meaning that at one tested value, the original and optimized versions did not have the same finite/NaN status.

### Large values of `x[1]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
879

Original vs Manual NaN mismatches:
879

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
50.15801236813767
Varied value:
15.89282865622978

Original vs Manual max finite difference:
50.15801236813767
Varied value:
15.89282865622978

Manual vs Herbie max finite difference:
0.0
```

The Manual and Herbie versions again produced identical results.

The original version had 879 NaN mismatches compared with both optimized versions. This indicates that the original computational form becomes unstable for many large values of `x[1]`.

The likely reason is that `s2` becomes extremely large:

```python
s2 = x[1] ** (x[1] * x[1] - cos(x[1]))
```

The original version first calculates this large value and later attempts to cancel it through:

```python
-s2 + t3
```

However, if `s2` overflows or loses important information before the cancellation occurs, the mathematically equivalent cancellation cannot recover the correct result.

The optimized expressions remove `s2` completely from the output calculation and therefore avoid this unnecessary intermediate value.

## First-Order Derivative Results

### Small values of `x[1]`

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
3.0
Varied value:
1.4916681462400365e-154

Original vs Manual max finite difference:
3.0
Varied value:
1.4916681462400365e-154

Manual vs Herbie max finite difference:
0.0
```

The Manual and Herbie first-order derivatives were identical.

The original version had 500 NaN mismatches compared with both optimized versions. This means that for approximately half of the small-number test cases, the original derivative contained at least one `NaN` or `Inf` value where the optimized versions produced a finite result.

The largest finite difference between the original and optimized derivatives was `3.0`.

### Large values of `x[1]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
879

Original vs Manual NaN mismatches:
879

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
3.0
Varied value:
4.79380849508911

Original vs Manual max finite difference:
3.0
Varied value:
4.79380849508911

Manual vs Herbie max finite difference:
0.0
```

The original first-order derivative had 879 NaN mismatches, while the Manual and Herbie derivatives remained identical.

This again shows that calculating `s2` before cancelling it causes significant numerical instability, especially as `x[1]` grows.

## Second-Order Derivative Results

### Small values of `x[1]`

```text
All three completely NaN/Inf:
1

Original vs Herbie NaN mismatches:
666

Original vs Manual NaN mismatches:
666

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
0.0
Varied value:
4.514686584930564e-103

Original vs Manual max finite difference:
0.0
Varied value:
4.514686584930564e-103

Manual vs Herbie max finite difference:
0.0
```

One test case produced only `NaN` or `Inf` values in all three versions.

The original version also had 666 NaN mismatches compared with both optimized versions.

The maximum finite difference was `0.0`. This does not contradict the large number of NaN mismatches. It means that whenever both versions had finite values at the same positions, those finite values were equal. The differences occurred in positions where one version was finite and the other was `NaN` or `Inf`.

### Large values of `x[1]`

```text
All three completely NaN/Inf:
0

Original vs Herbie NaN mismatches:
880

Original vs Manual NaN mismatches:
880

Herbie vs Manual NaN mismatches:
0

Original vs Herbie max finite difference:
0.0
Varied value:
1.0

Original vs Manual max finite difference:
0.0
Varied value:
1.0

Manual vs Herbie max finite difference:
0.0
```

The original second-order derivative had 880 NaN mismatches compared with both optimized versions.

As in the small-number sweep, the finite overlapping values were identical, which is why the maximum finite difference was `0.0`. However, the large number of NaN mismatches shows that the original version failed to produce finite derivatives in many cases where the optimized versions remained finite.

## Conclusion

The Manual and Herbie rewrites were mathematically equivalent and produced identical results in every experiment.

The original expression unnecessarily calculates:

```python
s2 = x[1] ** (x[1] * x[1] - cos(x[1]))
```

even though `s2` later cancels from the final output. This intermediate value can become extremely large and can overflow or introduce numerical instability before the cancellation takes place.

The optimization removes `s2` from the final output calculation entirely. This greatly improves numerical stability, especially for large values of `x[1]`.

The strongest evidence is the number of NaN mismatches:

```text
Small sweep:
Primal:       1
First order:  500
Second order: 666

Large sweep:
Primal:       879
First order:  879
Second order: 880
```

The Manual and Herbie versions had no NaN mismatches with each other and always produced identical finite results.

Therefore, this rewrite provides strong evidence of improved numerical stability. However, proving that the optimized versions are also more accurate would still require comparison against a higher-precision reference implementation.
