import pandas as pd
import numpy as np
import math

SOLVER_MILO = "highs"
SOLVER_MINLO = "ipopt"
BOUND = 0.3

from amplpy import AMPL

def cut_patterns_by_stock(width_s, weight_s, finish, patterns, BOUND_KEY):
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
        
        subject to feasible_pattern_max_margin:
          sum{p in P, f in F} b[p]* a[f,p] * width_f[f] >= 0.96 * width_s;
        
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
        
        subject to feasible_pattern_max_margin:
          sum{p in P, f in F} b[p]* a[f,p] * width_f[f] >= 0.96 * width_s;
        
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