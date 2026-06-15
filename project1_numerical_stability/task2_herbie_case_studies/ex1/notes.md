## Results and Discussion

The manual and Herbie transformations remove the unstable division-and-multiplication pattern from the original expression.

For the tested input, the manual and Herbie versions produced identical primal outputs. Small differences appeared in the first- and second-order derivatives.

Although the manual and Herbie expressions are mathematically equivalent, they use a different order of multiplication. Since floating-point multiplication is not perfectly associative, the different intermediate rounding can produce slightly different derivative values.

### Primal

```text
Manual vs Original: 1.7053025658242404e-13
Herbie vs Original: 1.7053025658242404e-13
Manual vs Herbie:   0.0
```

The manual and Herbie versions produced identical primal outputs for this input.

### First-order derivative



Manual vs Original: 9.094947017729282e-13
Herbie vs Original: 4.547473508864641e-13
Manual vs Herbie:   4.547473508864641e-13
```

For the first-order derivative, the Herbie result was closer to the original result than the manual result.

### Second-order derivative



Manual vs Original: 9.094947017729282e-13
Herbie vs Original: 1.8189894035458565e-12
Manual vs Herbie:   9.094947017729282e-13
```

For the second-order derivative, the manual result was closer to the original result than the Herbie result.

These pairwise differences show that the algebraically equivalent expressions have different floating-point behavior. However, they do not prove which version is the most accurate, because the original result is not a high-precision ground-truth reference.
