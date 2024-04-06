# Column Generation for the Cutting Stock Problem

This model is an example of the cutting stock problem. The Cutting Stock Problem deals with the problem of cutting 
stock material with the same, fixed width — such as paper rolls — into smaller pieces, according to a set of orders 
specifying both the widths and the demand requirements, so as to minimize the amount of wasted material. 
This problem is formulated as an integer programming problem using the Gurobi Python API and solved with a
decomposition technique called Delayed Column Generation using the Gurobi Optimizer.

This modeling example is at the advanced level, where we assume that you know Python and the Gurobi Python API and 
that you have advanced knowledge of building mathematical optimization models. Typically, the objective function 
and/or constraints of these examples are complex or require advanced features of the Gurobi Python API.

## The flow

- Step 1: Create all possible patterns for available stocks X required finished goods
- Step 2: Calculate the weight of each set pattern from cutting stock s (the weight of the finished goods depending on the weight of mother coil)
- Step 3: Constraint the weight of each item in pattern smaller than upper bound of weight 
        3.1 First try:
        Upper bound of weight
