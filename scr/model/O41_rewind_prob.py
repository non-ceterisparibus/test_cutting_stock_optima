import numpy as np

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpStatus

from O41_linear_solver import LinearProb
    
class RewindProb(LinearProb):
  # have condition to decide proportion of orginal MC to cut # apply for many FG and 1 stock MC
  # then move to the linear problem -> to decide the trim loss only
  def __init__(self, stock, finish):
    super().__init__(stock, finish)
    self.ratio = 0.5 # default ratio --> NEED TO CONSIDER SMALLEST LEFT OVER ALLOWED
    
  def _cut_partial_stock(self ):
    self.stock['weight'] *= self.ratio
    
  def ratio_array(self):
    
  def rewind_ratio(self):
    # xac dinh ratio da phai tinh den weight cat
    sum_demand = sum([self.finish[f]["need_cut"]for f in self.finish.keys()])
    sum_upper_demand = sum([self.finish[f]["need_cut"]for f in self.finish.keys()])
    
  
  def run(self):
    self.make_naive_patterns()
    probstt = self.optimize_cut()
    
    if probstt == "Optimal":
      return self.solution, self.over_cut
    else:
      return 
      
  
class SemiProb(LinearProb):
  def __init__(self, stock, finish):
    super().__init__(stock, finish)
  
  