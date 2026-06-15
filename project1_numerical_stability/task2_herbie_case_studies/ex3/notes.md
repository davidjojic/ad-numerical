Same as in the last example, both Herbie and the manual optimization gave exactly the same output. The only small difference is the order of addition, but as we can see, this did not cause any numerical difference.

As shown in the example below, when x[2] becomes smaller and smaller, the difference between the original and optimized functions becomes more visible.

PRIMAL
Manual vs Original:
145602194.40654054

Herbie vs Original:
145602194.40654054

Manual vs Herbie:
0.0
FIRST-ORDER DERIVATIVE
Shape: (10, 10)

Manual vs Original:
9.0e15

Herbie vs Original:
9.0e15

Manual vs Herbie:
0.0
SECOND-ORDER DERIVATIVE
Shape: (10, 10, 10)

Manual vs Original:
1.4e24

Herbie vs Original:
1.4e24

Manual vs Herbie:
0.0