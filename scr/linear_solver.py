### LOAD FINISH AND MOTHER COIL BY PARAMS - CUSTOMER
import pandas as pd
import numpy as np
import copy

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

def linear_program(finish, width_s, weight_s, MIN_MARGIN, BOUND_KEY, naive_patterns):
    
    """
    # upper bound with over-cut
        param f_upper_demand{F};
        param a_upper_bound{F};
        var a{f in F} integer >= 0, <= a_upper_bound[f];
      maximize total_width:
        sum{f in F} a[f] * width_f[f];
      subject to feasible_pattern_min_margin:
        sum{f in F} a[f] * width_f[f] <= width_s_min_margin;
      subject to feasible_pattern_max_margin:
        sum{f in F} a[f] * width_f[f] >= 0.96 * width_s;
      subject to weight_demand {f in F}:
          a[f] * width_f[f] * wu <= f_upper_demand[f];
    """
    
    # Create the problem
    prob = LpProblem("CuttingStock", LpMaximize)

    # Data and parameters
    F = list(finish.keys())
    width_s_min_margin = width_s - MIN_MARGIN
    width_f = {f: finish[f]["width"] for f in finish.keys()}
    wu = weight_s / width_s
    f_upper_demand = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}
    a_upper_bound = {f: max([naive_patterns[i]['cuts'][f] for i, _ in enumerate(naive_patterns)]) for f in finish.keys()}

    # Decision variables
    a = {f: LpVariable(f'a[{f}]', lowBound=0, upBound=a_upper_bound[f], cat='Integer') for f in F}

    # Objective function: maximize total width
    prob += lpSum(a[f] * width_f[f] for f in F), "TotalWidth"

    # Constraints
    # Feasible pattern min margin
    prob += lpSum(a[f] * width_f[f] for f in F) <= width_s_min_margin, "FeasiblePatternMinMargin"

    # Feasible pattern max margin
    prob += lpSum(a[f] * width_f[f] for f in F) >= 0.96 * width_s, "FeasiblePatternMaxMargin"

    # Weight demand
    for f in F:
        prob += a[f] * width_f[f] * wu <= f_upper_demand[f], f"WeightDemand_{f}"

    # Solve the problem
    prob.solve()

    # Check the result
    if prob.status == 1:
        print("Solution:")
        for f in F:
            print(f'[{f}] = {value(a[f])}')
        trim_loss = width_s - sum([finish[f]["width"] * value(a[f]) for f in finish.keys()])
        weight_loss = trim_loss * wu
        weight_cut = sum([finish[f]["width"] * value(a[f]) for f in finish.keys()]) * wu
        over_cut = sum([finish[f]["need_cut"] for f in finish.keys()]) + weight_cut
        # print('Total w =', value(prob.objective))
        # print(f"Trim Loss: {trim_loss}, Weight loss : {weight_loss}")

def solve_cutting_stock_ortools(finish, width_s, weight_s, MIN_MARGIN, BOUND_KEY, naive_patterns, SOLVER_MILO='SCIP'):
    
    """"
    # upper bound with over-cut
        param f_upper_demand{F};
        param a_upper_bound{F};
        var a{f in F} integer >= 0, <= a_upper_bound[f];
      maximize total_width:
        sum{f in F} a[f] * width_f[f];
      subject to feasible_pattern_min_margin:
        sum{f in F} a[f] * width_f[f] <= width_s_min_margin;
      subject to feasible_pattern_max_margin:
        sum{f in F} a[f] * width_f[f] >= 0.96 * width_s;
      subject to weight_demand {f in F}:
          a[f] * width_f[f] * wu <= f_upper_demand[f];
    """
    # Create the solver
    solver = pywraplp.Solver.CreateSolver(SOLVER_MILO)

    # Data and parameters
    F = list(finish.keys())
    width_s_min_margin = width_s - MIN_MARGIN
    width_f = {f: finish[f]["width"] for f in finish.keys()}
    wu = weight_s / width_s
    f_upper_demand = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}
    a_upper_bound = {f: max([naive_patterns[i]['cuts'][f] for i, _ in enumerate(naive_patterns)]) for f in finish.keys()}

    # Decision variables
    a = {f: solver.IntVar(0, a_upper_bound[f], f'a[{f}]') for f in F}

    # Objective function: maximize total width
    solver.Maximize(solver.Sum(a[f] * width_f[f] for f in F))

    # Constraints
    # Feasible pattern min margin
    solver.Add(solver.Sum(a[f] * width_f[f] for f in F) <= width_s_min_margin)

    # Feasible pattern max margin
    solver.Add(solver.Sum(a[f] * width_f[f] for f in F) >= 0.96 * width_s)

    # Weight demand
    for f in F:
        solver.Add(a[f] * width_f[f] * wu <= f_upper_demand[f])

    # Solve the problem
    status = solver.Solve()

    # Check the result
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        for f in F:
          print(f'[{f}] = {a[f].solution_value()}')
        trim_loss = width_s - sum([finish[f]["width"]*a[f].solution_value() for f in finish.keys()])
        weight_loss = trim_loss * wu
        weight_cut = sum([finish[f]["width"]*a[f].solution_value() for f in finish.keys()]) * wu
        over_cut = sum([finish[f]["need_cut"] for f in finish.keys()]) + weight_cut
        # print('Total w =', solver.Objective().Value())
        # print(f"Trim Loss: {trim_loss}, Weigh loss : {weight_loss}")
        return {f'trim_loss: {trim_loss}, over_cut: {over_cut},weight_loss: {weight_loss}'}
    else:
        print("The problem does not have an optimal solution.")