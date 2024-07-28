### LOAD FINISH AND MOTHER COIL BY PARAMS - CUSTOMER
import pandas as pd
import numpy as np

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpStatus
from O31_steel_objects import FinishObjects, StockObjects

# DEFINE PROBLEM
class LinearProb:
  def __init__(self, stock, finish):
    self.stock = stock # single stock 
    self.finish = finish
    self.over_cut = None
    self.solution = {}
  
  def make_naive_patterns(self):
    """
    Generates patterns of feasible cuts from stock width to meet specified finish widths. not considering the weight contraint
    """
    self.patterns = []
    for f in self.finish:
        feasible = False
        # max number of f that fit on s, bat buoc phai round down vi ko cat qua width duoc
        num_cuts_by_width = int((self.stock[0]["width"]-self.stock[0]["min_margin"]) / self.stock[f]["width"])
        # max number of f that satisfied the need cut WEIGHT BOUND
        # num_cuts_by_weight = round((self.finish[f]["upper_bound"] * self.stock[0]["width"] ) / (self.stock[f]["width"] * self.stock[0]['weight']))
        # min of two max will satisfies both
        # num_cuts = min(num_cuts_by_width, num_cuts_by_weight)
        
        # make pattern and add to list of patterns
        if num_cuts_by_width > 0:
          feasible = True
          cuts_dict = {key: 0 for key in self.finish.keys()}
          cuts_dict[f] = num_cuts_by_width
          trim_loss = self.stock[0]['width'] - sum([self.finish[f]["width"] * cuts_dict[f] for f in self.finish.keys()])
          trim_loss_pct = round(trim_loss/self.stock[0]['width'] * 100, 3)
          self.patterns.append({"cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct })
            
        if not feasible :
            pass
            
  def optimize_cut(self):
    # Create the problem
    prob = LpProblem("CuttingStock", LpMaximize)

    # Data and parameters
    F = list(self.finish.keys())
    width_s_min_margin = self.stock[0]['width'] - self.stock[0]['min_margin']
    width_f = {f: self.finish[f]["width"] for f in self.finish.keys()}
    wu = self.stock[0]['weight'] / self.stock[0]['width']
    # f_upper_demand = {f: self.finish[f][f"upper_bound"] for f in self.finish.keys()}
    # f_demand = {f: self.finish[f][f"need_cut"] for f in self.finish.keys()}
    a_upper_bound = {f: max([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns)]) for f in self.finish.keys()}

    # Decision variables
    a = {f: LpVariable(f'a[{f}]', lowBound=0, upBound=a_upper_bound[f], cat='Integer') for f in F}

    # Objective function: maximize total width
    prob += lpSum(a[f] * width_f[f] for f in F), "TotalWidth"

    # Constraints
    # Feasible pattern min margin
    prob += lpSum(a[f] * width_f[f] for f in F) <= width_s_min_margin, "FeasiblePatternMinMargin"

    # Feasible pattern max margin
    # prob += lpSum(a[f] * width_f[f] for f in F) >= 0.96 * self.stock[0]['width'], "FeasiblePatternMaxMargin"

    # Weight demand
    # for f in F:
      # prob += a[f] * width_f[f] * wu <= f_upper_demand[f], f"WeightDemand_{f}"
      # prob += a[f] * width_f[f] * wu > f_demand[f], f"Demand_{f}"
      
    # Solve the problem
    prob.solve()

    # Check the result
    if prob.status == "Optimal":
      self.solution['cuts'] = {}
      self.solution['cut_w'] = {}
      self.solution['trim_loss'] = self.stock[0]['width'] - sum([self.finish[f]["width"] * value(a[f]) for f in self.finish.keys()])
      for f in F:
        self.solution['cuts'][f] = value(a[f])
        self.solution['cut_w'][f] = self.finish[f]["width"] * value(a[f]) * wu
          
      self.over_cut = [self.finish[f]["need_cut"] - self.solution['cut_w'][f] for f in self.finish.keys()]
    
    return prob.status # overcut, solution co trong self
  
  def run(self):
    self.make_naive_patterns()
    prostt = self.optimize_cut()
    
    return prostt
