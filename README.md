# Column Generation for the Cutting Stock Problem

This model is an example of the cutting stock problem. The Cutting Stock Problem deals with the problem of cutting 
stock material with the same, fixed width — such as paper rolls — into smaller pieces, according to a set of orders 
specifying both the widths and the demand requirements, so as to minimize the amount of wasted material. 
This problem is formulated as an integer programming problem using the Gurobi Python API and solved with a
decomposition technique called Delayed Column Generation using the Gurobi Optimizer.

This modeling example is at the advanced level, where we assume that you know Python and the Gurobi Python API and 
that you have advanced knowledge of building mathematical optimization models. Typically, the objective function 
and/or constraints of these examples are complex or require advanced features of the Gurobi Python API.

## The Optima Flow with Coil
User inputs hyperparameter for cutting -> The-Pool
Then, 

- Step 0: Recount and filter the set going to Step 1 <br/>
0.0 Picking the PO Batch from The-Pool PO :
Current technical cutting ability of warehouse is not limited by the number of choosen finished goods type.
However, current format has upto 10 (12) kinds of finished goods that would be sent to warehouse.
So, we will create a cutoff on the number of FG kind - 10 (12) - a PO batch
Threshold 1:
Selecting need cut to add to batch <br/>
0.5% (finished goods ratio) OR around  X0.0 KGs ( <10 KGs or 0.5% which comes first)
Considering criteria to pick a PO batch?? <br/>
** By width: the larger, the more prioritied <br/>
** By weight (need-cut): the larger, the more prioritied. 

0.1 Picking the Mother Coil set: <br/>
Filter the set of MC that has the weight greater or equal total weight of set PO

Considering criteria to pick a Mother Coil <br/>
** By width: the larger, the more prioritied

- Step 1: The Possible Space <br/>
Create all possible patterns for available stocks (width) X required finished goods (width)
Note: with each possible pattern we has each possible finished goods weight (cut from MC)
- Step 2: The Set of Contraints <br/>
Threshold 2: to set the constraints
Calculate the weight of each set pattern from cutting stock s (the weight of the finished goods depending on the weight of choosen Mother Coil)
- Step 3: The Contraints  (on each Cus-Spec) <br/>
Constraint the weight of each item in pattern smaller than upper bound of weight. Can be relaxed independently on each contraint <br/> (dynamic)
        3.1 First try: <br/>
        Upper bound of weight = Need_Cut + 10% Need_cut<br/>
        3.2 Second try:<br/>
        Upper bound of weight = Need_Cut + 50% FC1<br/>
        3.3 Third try:<br/>
        Upper bound of weight = Need_Cut + 200% FC1<br/>

After achieving the optimal, the remaining need_cut (from previous batch) will be added to The-Pool and selected for the optimization again.

Stop Point: All Required Finished Goods supposed to be cut have to need_cut < 50


AMPL on Docker : https://dev.ampl.com/ampl/docker/index.html


https://github.com/ampl/colab.ampl.com/blob/master/authors/fdabrandao/examples/pattern_generation.ipynb