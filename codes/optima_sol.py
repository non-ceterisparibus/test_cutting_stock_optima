import pandas as pd
import numpy as np
import math

SOLVER_MILO = "highs"
SOLVER_MINLO = "ipopt"
BOUND = 0.3

from amplpy import AMPL
from ortools.linear_solver import pywraplp

def cut_stock_by_patterns(width_s, weight_s, finish, patterns, BOUND_KEY, MIN_MARGIN):
    m = AMPL()
    m.eval("reset data;")
    m.eval(
    """
        set F;
        set P;

        # width stock
        param width_s integer;
        # width stock minus min_margin
        param width_s_min_margin integer;
        # weight per unit of stock
        param wu > 0;
        # width finished pieces
        param width_f{F};

        # upper bound with over-cut
        param f_upper_demand{F};
        # param demand_finish{F};
        # how many f pieces are returned from pattern p
        param a{F, P};
        # which stock s is choosen for pattern p
        var b{P} binary;

        # Find the patterns of stock that minimize the loss
        minimize trim_loss:
          width_s - sum{p in P, f in F} b[p] * a[f,p] * width_f[f];
        
        subject to assign_each_finish_to_pattern:
          sum{p in P} b[p]  = 1;
        
        subject to feasible_pattern_max_margin:
          sum{p in P, f in F} b[p]* a[f,p] * width_f[f] >= 0.96 * width_s;
        
        subject to feasible_pattern_min_margin:
          sum{p in P, f in F}  b[p] * a[f,p] * width_f[f] <= width_s_min_margin;

        subject to weight_demand {f in F}:
          sum{p in P} b[p] * a[f,p] * width_f[f] * wu <= f_upper_demand[f];
    """
    )
    m.set["F"] = list(finish.keys())
    m.set["P"] = list(range(len(patterns)))

    m.param["width_s"] = width_s
    m.param["width_s_min_margin"] = width_s - MIN_MARGIN
    m.param["width_f"] = {f: finish[f]["width"] for f in finish.keys()}

    m.param["wu"] = weight_s/width_s  # stock weight per unit (unique)
    # m.param["demand_finish"] = {f: finish[f]["need_cut"] for f in finish.keys()}
    m.param["f_upper_demand"] = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}

    a = {
        (f, p): patterns[p]["cuts"][f]
        for p in range(len(patterns))
        for f in finish.keys()
    }
    m.param["a"] = a
    
    m.option["solver"] = SOLVER_MILO
    m.get_output("solve;")

    opt_patterns = [p for p in range(len(patterns)) if m.var["b"][p].value() > 0]
    return opt_patterns

def solve_cutting_stock_ampl(width_s, weight_s, finish, naive_patterns, MIN_MARGIN,BOUND_KEY):
    m = AMPL()
    m.eval("reset data;")
    m.eval(
    """
        reset;
        set F;

        # width stock
        param width_s integer;
        # width stock minus min_margin
        param width_s_min_margin integer;
        # weight per unit of stock
        param wu > 0;
        # width finished pieces
        param width_f{F};
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
    )
    m.set["F"] = list(finish.keys())
    # m.set["P"] = list(range(len(finish)))  # neu range nay ko du lon thi ko tim duoc pattern
    m.param["width_s"] = width_s
    m.param["width_s_min_margin"] = width_s - MIN_MARGIN
    m.param["width_f"] = {f: finish[f]["width"] for f in finish.keys()}
    m.param["wu"] = weight_s /width_s # stock weight per unit (unique)
    m.param["f_upper_demand"] = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}
    a_upper_bound = {f: max([naive_patterns[i]['cuts'][f] for i,_ in enumerate(naive_patterns)]) for f in finish.keys()}
    m.param["a_upper_bound"] = a_upper_bound

    m.option["solver"] = SOLVER_MILO
    m.get_output("solve;")

    # opt_patterns = [p for p in range(len(finish)) if m.var["b"][p].value() > 0]
    opt_patterns = {f: m.var["a"][f].value() for f in finish.keys()}
    return opt_patterns

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
    if not solver:
        print("Solver not found.")
        return

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
  

# error and defective 
def cut_patterns_by_defective_stock(width_s, weight_s, finish, patterns, BOUND_KEY):
    m = AMPL()
    m.eval("reset data;")
    m.eval(
    """
        set F;
        set P;

        # width stock
        param width_s integer;
        # weight per unit of stock
        param wu > 0;
        # width finished pieces
        param width_f{F};

        # upper bound with over-cut
        param f_upper_demand{F};
        param demand_finish{F};
        # how many f pieces are returned from pattern p
        param a{F, P};
        # which stock s is choosen for pattern p
        var b{P} binary;

        # Find the patterns of stock that minimize the loss
        minimize trim_loss:
          width_s - sum{p in P, f in F} b[p] * a[f,p] * width_f[f];
        
        subject to assign_each_finish_to_pattern:
          sum{p in P} b[p]  = 1;
        
        # subject to feasible_pattern_max_margin:
        #   sum{p in P, f in F} b[p]* a[f,p] * width_f[f] >= 0.96 * width_s;
        
        subject to feasible_pattern_min_margin:
          sum{p in P, f in F}  b[p] * a[f,p] * width_f[f] <= width_s - 8;

        subject to weight_demand {f in F}:
          sum{p in P} b[p] * a[f,p] * width_f[f] * wu <= f_upper_demand[f];
    """
    )
    m.set["F"] = list(finish.keys())
    m.set["P"] = list(range(len(patterns)))

    m.param["width_s"] = width_s #stocks[s]["width"] 
    m.param["width_f"] = {f: finish[f]["width"] for f in finish.keys()}

    m.param["wu"] = weight_s/width_s  # stock weight per unit (unique)
    m.param["demand_finish"] = {f: finish[f]["need_cut"] for f in finish.keys()}
    m.param["f_upper_demand"] = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}

    a = {
        (f, p): patterns[p]["cuts"][f]
        for p in range(len(patterns))
        for f in finish.keys()
    }
    m.param["a"] = a
    
    m.option["solver"] = SOLVER_MILO
    m.get_output("solve;")

    opt_patterns = [p for p in range(len(patterns)) if m.var["b"][p].value() > 0]
    return opt_patterns