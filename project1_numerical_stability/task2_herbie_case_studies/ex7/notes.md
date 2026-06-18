## Example 7: Cancellation Around `atan2`

### Original expression

```cpp
s0 = ((x[3] + atan2(x[3], x[3])) - x[3]);
```

The corresponding Herbie input was:

```lisp
(- (+ x3 (atan2 x3 x3)) x3)
```

Herbie produced:

```lisp
(atan2 x3 x3)
```

Therefore, both the manual and Herbie versions use:

```cpp
s0 = atan2(x[3], x[3]);
```

### Mathematical simplification

In exact arithmetic, the `x[3]` and `-x[3]` terms cancel:

```text
(x[3] + atan2(x[3], x[3])) - x[3]
=
atan2(x[3], x[3])
```

For `x[3] != 0`, this expression is constant on each side of zero:

```text
atan2(x[3], x[3]) = pi / 4      when x[3] > 0

atan2(x[3], x[3]) = -3pi / 4    when x[3] < 0
```

The expression is not well-defined at `x[3] = 0`, so zero was excluded from the generated tests.

### Numerical root cause

The original expression is numerically unstable for large-magnitude values of `x[3]`.

For a large positive input, the computation is approximately:

```text
(x[3] + pi / 4) - x[3]
```

In binary64 arithmetic, when `x[3]` is sufficiently large, `pi / 4` is smaller than the spacing between adjacent representable numbers near `x[3]`.

Consequently:

```text
fl(x[3] + pi / 4) = x[3]
```

The following subtraction therefore evaluates to:

```text
fl(x[3] - x[3]) = 0
```

even though the exact result is `pi / 4`.

Thus, the original implementation may compute `s0` as zero, whereas the manual and Herbie implementations preserve the correct `atan2` value.

### Effect on derivatives

For `x[3] != 0`, the exact derivative of

```text
atan2(x[3], x[3])
```

is zero on each branch.

Using the partial derivatives of `atan2`:

```text
d/da atan2(a, b) = b / (a*a + b*b)
```

and

```text
d/db atan2(a, b) = -a / (a*a + b*b)
```

and substituting:

```text
a = x[3]
b = x[3]
```

the two chain-rule contributions cancel.

However, the large derivative differences observed in the experiment are not primarily caused by imperfect accumulation of these two contributions.

They arise because the original expression produces an incorrect floating-point primal value for `s0`.

The rest of the program contains nonlinear expressions and divisions that depend on `s0`, either directly or through later intermediate variables.

Therefore, changing `s0` from its correct `atan2` value to zero changes the local derivatives evaluated throughout the remaining computational graph.

This error is amplified in the Jacobian and even more strongly in the second-order derivatives.

### Experimental results

For values near zero, all three implementations produced identical primal, first-order, and second-order results.

For large-magnitude inputs, the original implementation differed substantially from both improved versions.

The maximum observed differences were approximately:

* primal output: `2.72e+39`
* first-order derivatives: `2.18e+40`
* second-order derivatives: `5.44e+41`

The manual and Herbie implementations were identical for every tested input:

* manual versus Herbie primal difference: `0`
* manual versus Herbie first-order difference: `0`
* manual versus Herbie second-order difference: `0`

These results show that removing the unstable addition-subtraction sequence preserves the mathematically intended value and prevents the resulting instability from being amplified through first- and second-order automatic differentiation.
