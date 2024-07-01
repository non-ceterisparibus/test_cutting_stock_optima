import pandas as pd
import numpy as np
import math

SOLVER_MILO = "highs"
SOLVER_MINLO = "ipopt"
BOUND = 0.3

from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value
from ortools.linear_solver import pywraplp

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

def cut_patterns(stocks, finish, patterns):
    # Create a LP minimization problem
    prob = LpProblem("PatternCuttingProblem", LpMinimize)

    # Define variables
    x = {p: LpVariable(f"x_{p}", lowBound=0, cat='Integer') for p in range(len(patterns))}

    # Objective function: minimize total cost
    prob += lpSum(1 * x[p] for p in range(len(patterns))), "TotalCost"

    # Constraints: meet demand for each finished part
    for f in finish:
        prob += lpSum(patterns[p]['cuts'][f] * x[p] for p in range(len(patterns))) >= finish[f]['demand_slice'], f"DemandSlice{f}"

    # Solve the problem
    prob.solve()

    # Extract results
    solution = [x[p].varValue for p in range(len(patterns))]
    total_cost = sum(1 * solution[p] for p in range(len(patterns)))

    return solution, total_cost

# NEW PATTERN
def new_pattern_problem(finish, width_s, ap_upper_bound, demand_duals):
    prob = LpProblem("NewPatternProblem", LpMaximize)

    # Decision variables - Pattern
    ap = {f: LpVariable(f"ap_{f}", 0, ap_upper_bound[f], cat="Integer") for f in finish.keys()}

    # Objective function
    # maximize marginal_cost:
        #    sum{f in F} ap[f] * demand_dual[f];
    prob += lpSum(ap[f] * demand_duals[f] for f in finish.keys()), "MarginalCut"

    # Constraints
    # subject to stock_length:
        #    sum{f in F} ap[f] * length_f[f] <= length_s;
    prob += lpSum(ap[f] * finish[f]["width"] for f in finish.keys()) <= width_s, "StockLength"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

    marg_cost = value(prob.objective)
    pattern = {f: int(ap[f].varValue) for f in finish.keys()}
    return marg_cost, pattern
# DUALITY
def generate_pattern_dual(stocks, finish, patterns):
    prob = LpProblem("GeneratePatternDual", LpMinimize)

    # Sets
    F = list(finish.keys())
    P = list(range(len(patterns)))

    # Parameters
    s = {p: patterns[p]["stock"] for p in range(len(patterns))}
    # c = {p: stocks[s[p]]["cost"] for p in range(len(patterns))}
    
    a = {(f, p): patterns[p]["cuts"][f] for p in P for f in F}
    demand_finish = {f: finish[f]["demand_slice"] for f in F}

    # Decision variables
    # var x{P} >= 0; # relaxed integrality
    x = {p: LpVariable(f"x_{p}", 0, None, cat="Continuous") for p in P}

    # Objective function
    # minimize stock used:
    #         sum{p in P} x[p];
    prob += lpSum(x[p] for p in P), "Cost"

    # Constraints
    for f in F:
        # sum{p in P} a[f,p]*x[p] >= demand_finish[f];
        prob += lpSum(a[f, p] * x[p] for p in P) >= demand_finish[f], f"Demand_{f}"
        # sum{p in P} a[f,p]*x[p] <= upper_demand_finish[f];

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

    # Extract dual values
    dual_values = {f: prob.constraints[f"Demand_{f}"].pi for f in F}

    ap_upper_bound = {
        f: max([int(stocks[s]["width"] / finish[f]["width"]) for s in stocks.keys()])
        for f in F
    }
    demand_duals = {f: dual_values[f] for f in F}

    marginal_values = {}
    pattern = {}
    for s in stocks.keys():
        marginal_values[s], pattern[s] = new_pattern_problem(
            finish, stocks[s]["width"], ap_upper_bound, demand_duals
        )

    s = max(marginal_values, key=marginal_values.get)
    new_pattern = {"stock": s, "cuts": pattern[s]}
    return new_pattern