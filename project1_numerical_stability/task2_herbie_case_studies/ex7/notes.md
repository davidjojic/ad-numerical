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

[
(x_3+\operatorname{atan2}(x_3,x_3))-x_3
=======================================

\operatorname{atan2}(x_3,x_3).
]

For (x_3 \neq 0), this expression is constant on each side of zero:

[
\operatorname{atan2}(x_3,x_3)
=============================

\begin{cases}
\pi/4, & x_3>0,\
-3\pi/4, & x_3<0.
\end{cases}
]

The expression is not well-defined at (x_3=0), so zero was excluded from the generated tests.

### Numerical root cause

The original expression is numerically unstable for large-magnitude values of (x_3).

For a large positive input, the computation is approximately:

[
(x_3+\pi/4)-x_3.
]

In binary64 arithmetic, when (x_3) is sufficiently large, the constant (\pi/4) is smaller than the spacing between adjacent representable numbers near (x_3). Consequently,

[
\mathrm{fl}(x_3+\pi/4)=x_3.
]

The following subtraction therefore evaluates to:

[
\mathrm{fl}(x_3-x_3)=0,
]

even though the exact result is (\pi/4).

Thus, the original implementation may compute `s0` as zero, whereas the manual and Herbie implementations preserve the correct `atan2` value.

### Effect on derivatives

For (x_3\neq0), the exact derivative of

[
\operatorname{atan2}(x_3,x_3)
]

is zero on each branch. Using the partial derivatives of `atan2`,

[
\frac{\partial}{\partial a}\operatorname{atan2}(a,b)
====================================================

\frac{b}{a^2+b^2},
]

[
\frac{\partial}{\partial b}\operatorname{atan2}(a,b)
====================================================

-\frac{a}{a^2+b^2},
]

and substituting (a=b=x_3), the two chain-rule contributions cancel.

However, the large derivative differences observed in the experiment are not primarily caused by imperfect accumulation of these two contributions. They arise because the original expression produces an incorrect floating-point primal value for `s0`.

The rest of the program contains nonlinear expressions and divisions that depend on `s0`, either directly or through later intermediate variables. Therefore, changing `s0` from its correct `atan2` value to zero changes the local derivatives evaluated throughout the remaining computational graph. This error is amplified in the Jacobian and even more strongly in the second-order derivatives.

### Experimental results

For values near zero, all three implementations produced identical primal, first-order, and second-order results.

For large-magnitude inputs, the original implementation differed substantially from both improved versions. The maximum observed differences were approximately:

* primal output: (2.72\times10^{39});
* first-order derivatives: (2.18\times10^{40});
* second-order derivatives: (5.44\times10^{41}).

The manual and Herbie implementations were identical for every tested input:

* manual versus Herbie primal difference: `0`;
* manual versus Herbie first-order difference: `0`;
* manual versus Herbie second-order difference: `0`.

These results show that removing the unstable addition-subtraction sequence preserves the mathematically intended value and prevents the resulting instability from being amplified through first- and second-order automatic differentiation.
