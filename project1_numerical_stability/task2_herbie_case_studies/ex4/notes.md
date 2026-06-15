In this example, both the manual optimization and Herbie produced mathematically equivalent functions. The only difference is the order of addition, but for the tested input this did not cause any numerical difference, as shown in the results below.

### PRIMAL
Manual vs Original:
30.1107537000258

Herbie vs Original:
30.1107537000258

Manual vs Herbie:
0.0


### FIRST-ORDER DERIVATIVE

Shape: (10, 10)

Manual vs Original:
1.593861581306106

Herbie vs Original:
1.593861581306106

Manual vs Herbie:
0.0

### SECOND-ORDER DERIVATIVE


Shape: (10, 10, 10)

Manual vs Original:
0.6960123970693185

Herbie vs Original:
0.6960123970693185

Manual vs Herbie:
0.0

Since the difference between the manual and Herbie versions is zero for the primal, first-order derivative, and second-order derivative, we can conclude that both optimizations behaved identically for this test input.
