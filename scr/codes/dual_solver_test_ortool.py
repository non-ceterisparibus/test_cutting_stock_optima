import pandas as pd
import numpy as np
import copy
# from model import FinishObjects, StockObjects
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, CPLEX_PY, PULP_CBC_CMD, GLPK ,GLPK_CMD, LpBinary, value
from ortools.linear_solver import pywraplp

# DEFINE PROBLEM
class testDualProblemOR:
    def __init__(self, finish, stocks):
        self.len_stocks = len(stocks)
        self.dual_finish = finish
        self.dual_stocks = stocks
        self.start_stocks = stocks
        self.start_finish = finish
        self.final_solution_patterns = []

    # PHASE 1: Naive/ Dual Pattern Generation
    def _make_naive_patterns(self):
        """
        Generates patterns of feasible cuts from stock width to meet specified finish widths.
        """
        self.patterns = []
        for f in self.dual_finish:
            feasible = False
            for s in self.dual_stocks:
                # max number of f that fit on s, bat buoc phai round down vi ko cat qua width duoc
                num_cuts_by_width = int((self.dual_stocks[s]["width"]-self.dual_stocks[s]["min_margin"]) / self.dual_finish[f]["width"])
                # max number of f that satisfied the need cut WEIGHT BOUND
                num_cuts_by_weight = round((self.dual_finish[f]["upper_bound"] * self.dual_stocks[s]["width"] ) / (self.dual_finish[f]["width"] * self.dual_stocks[s]['weight']))
                # min of two max will satisfies both
                num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

                # make pattern and add to list of patterns
                if num_cuts > 0:
                    feasible = True
                    cuts_dict = {key: 0 for key in self.dual_finish.keys()}
                    cuts_dict[f] = num_cuts
                    trim_loss = self.dual_stocks[s]['width'] - sum([self.dual_finish[f]["width"] * cuts_dict[f] for f in self.dual_finish.keys()])
                    trim_loss_pct = round(trim_loss/self.dual_stocks[s]['width'] * 100, 3)
                    self.patterns.append({"stock":s, "inventory_id": s,
                                          'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct ,
                                          "explanation":"",'remark':"","cutting_date":"",
                                          "stock_weight": self.dual_stocks[s]['weight'], 'stock_width':self.dual_stocks[s]['width'],
                                          "cuts": cuts_dict,
                                          "details": [{'order_no': f, 'width':self.dual_finish[f]['width'], 'lines': cuts_dict[f]} for f in cuts_dict.keys()] 
                    })
                    
            if not feasible:
                continue
                # print(f"No feasible pattern was found for Stock {s} and FG {f}")

    def create_finish_demand_by_line_w_naive_pattern(self):
        self._make_naive_patterns()
        dump_ls = {}
        for f, finish_info in self.dual_finish.items():
            try:
                non_zero_min = min([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns) if self.patterns[i]['cuts'][f] != 0])
            except ValueError:
                non_zero_min = 0
                
            try:    
                dump_ls[f] = {**finish_info
                            ,"upper_demand_line": max([self.patterns[i]['cuts'][f] for i,_ in enumerate(self.patterns)])
                            ,"demand_line": non_zero_min }
            except ValueError:
                dump_ls[f] = {**finish_info
                            ,"upper_demand_line":non_zero_min+1
                            ,"demand_line": non_zero_min }
                
       
        # Filtering the dictionary to include only items with keys in keys_to_keep
        self.dual_finish = {k: v for k, v in dump_ls.items() if v['upper_demand_line'] > 0} # xem lai dieu kien nay, tuc la neu cat dai nay voi stock hien co thì overcut lon
    
    # PHASE 2: Pattern Duality
    def _filter_out_overlapped_stock(self):
        """
        Find stocks {stock:receiving_date,width, weight, qty} 
        with condition, take the list of pattern diff from the key
        """
        filtered_list = {}
        for s, stock_info in self.dual_stocks.items():
            if s != self.max_key:
                filtered_list[s] = {**stock_info}
            
        self.dual_stocks = copy.deepcopy(filtered_list)

    def _count_pattern(self,patterns):
        """
        Count each stock is used how many times
        """

        stock_counts = {}

        # Iterate through the list and count occurrences of each stock
        for item in patterns:
            stock = item['stock']
            count = 1
            if stock in stock_counts:
                stock_counts[stock] += count
            else:
                stock_counts[stock] = count

        return stock_counts

    def _new_pattern_problem(self, width_s, ap_upper_bound, demand_duals, MIN_MARGIN):
        # Create the solver instance (CBC Solver)
        solver = pywraplp.Solver.CreateSolver('CBC')

        # Decision variables - Pattern
        ap = {f: solver.IntVar(0, ap_upper_bound[f], f"ap_{f}") for f in self.dual_finish.keys()}

        # Objective function: maximize marginal_cut
        objective = solver.Objective()
        for f in self.dual_finish.keys():
            objective.SetCoefficient(ap[f], demand_duals[f])
        objective.SetMaximization()

        # Constraints - subject to stock_width - MIN MARGIN
        constraint_min_margin = solver.Constraint(-solver.infinity(), width_s - MIN_MARGIN)
        for f in self.dual_finish.keys():
            constraint_min_margin.SetCoefficient(ap[f], self.dual_finish[f]["width"])

        # Constraints - subject to trim loss 4%
        constraint_trim_loss = solver.Constraint(0.96 * width_s, solver.infinity())
        for f in self.dual_finish.keys():
            constraint_trim_loss.SetCoefficient(ap[f], self.dual_finish[f]["width"])

        # Solve the problem
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            marg_cost = solver.Objective().Value()
            pattern = {f: int(ap[f].solution_value()) for f in self.dual_finish.keys()}
        else:
            marg_cost = None
            pattern = None

        return marg_cost, pattern
    
    def _generate_dual_pattern(self):
        # Create the solver instance (CBC Solver)
        solver = pywraplp.Solver.CreateSolver('CBC')

        # Sets
        F = list(self.dual_finish.keys())
        P = list(range(len(self.patterns)))

        # Parameters
        s = {p: self.patterns[p]["stock"] for p in P}
        a = {(f, p): self.patterns[p]["cuts"][f] for p in P for f in F}
        demand_finish = {f: self.dual_finish[f]["demand_line"] for f in F}
        upper_demand_finish = {f: self.dual_finish[f]["upper_demand_line"] for f in F}

        # Decision variables (relaxed integrality)
        x = {p: solver.NumVar(0, 20, f"x_{p}") for p in P}

        # OBJECTIVE function: minimize stock used
        objective = solver.Objective()
        for p in P:
            objective.SetCoefficient(x[p], 1)
        objective.SetMinimization()

        # Constraints
        for f in F:
            # Lower bound constraint for demand
            demand_constraint = solver.Constraint(demand_finish[f], solver.infinity())
            for p in P:
                demand_constraint.SetCoefficient(x[p], a[f, p])

            # Upper bound constraint for demand
            upper_demand_constraint = solver.Constraint(-solver.infinity(), upper_demand_finish[f])
            for p in P:
                upper_demand_constraint.SetCoefficient(x[p], a[f, p])

        # Solve the problem
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            # Extract dual values (shadow prices)
            dual_values = {f: demand_constraint.dual_value() for f in F}

            ap_upper_bound = {
                f: max([self.patterns[i]['cuts'][f] for i in P]) for f in self.dual_finish.keys()
            }
            demand_duals = {f: dual_values[f] for f in F}

            marginal_values = {}
            pattern = {}
            for stock in self.dual_stocks.keys():
                width_s = self.dual_stocks[stock]["width"]
                min_margin = self.dual_stocks[stock]["min_margin"]
                marginal_values[stock], pattern[stock] = self._new_pattern_problem(
                    width_s, ap_upper_bound, demand_duals, min_margin
                )

            try:
                selected_stock = max(marginal_values, key=marginal_values.get)  # Pick the best stock
                cuts_dict = pattern[selected_stock]

                new_pattern = {
                    "stock": selected_stock,
                    "inventory_id": selected_stock,
                    "explanation": "",
                    "remark": "",
                    "cutting_date": "",
                    'stock_weight': self.dual_stocks[selected_stock]['weight'],
                    'stock_width': self.dual_stocks[selected_stock]["width"],
                    "cuts": cuts_dict,
                    "details": [
                        {
                            'order_no': f,
                            'width': self.dual_finish[f]['width'],
                            'lines': cuts_dict[f]
                        } for f in cuts_dict.keys()
                    ]
                }
            except ValueError:
                new_pattern = None

            return new_pattern
        else:
            return None
    
    # Solve PATTERN Duality
    def generate_patterns(self):
        n = 1
        remove_stock = True
        self.max_key = None
        while remove_stock == True:
            self._filter_out_overlapped_stock()
            new_pattern = self._generate_dual_pattern() 
            dual_pat = []
            while (new_pattern not in dual_pat) and (new_pattern is not None):
                self.patterns.append(new_pattern)   
                dual_pat.append(new_pattern)        # dual pat de tinh stock bi lap nhieu lan
                new_pattern = self._generate_dual_pattern()

            # filter stock having too many patterns
            if not dual_pat:
                remove_stock = False
            else:
                ls = self._count_pattern(dual_pat)
                self.max_key = max(ls, key=ls.get) 
                max_count = ls[self.max_key]
                if max_count >= 1 and n < self.len_stocks: #remove den khi chi con 1 stock???
                    remove_stock = True
                    n +=1
                else: 
                    remove_stock = False

    # PHASE 3: Filter Patterns
    def _count_fg_cut(self, dict):
        count = sum(1 for value in dict.values() if value > 0)
        return count
    
    def filter_patterns_and_stocks_by_constr(self):
        """_summary_
        {"stock":s, "inventory_id": s,
        "explanation":"",'remark':"","cutting_date":"",
            'stock_weight': 4000, 
            'stock_width': 1219,
            "cuts": pattern[s],
            "details": [{'order_no': f, 
                         'width':self.dual_finish[f]['width'], 
                         'lines': cuts_dict[f]} 
                        for f in cuts_dict.keys()] 
            }
        """
        # Initiate list
        self.filtered_patterns = []

        # Filter patterns
        for pattern in self.patterns:
            cuts_dict= pattern['cuts']
            count_fg_cut = self._count_fg_cut(cuts_dict)
            width_s = self.start_stocks[pattern['stock']]['width']
            trim_loss = width_s - sum([self.start_finish[f]["width"] * cuts_dict[f] for f in cuts_dict.keys()])
            trim_loss_pct = round(trim_loss/width_s * 100, 3)
            if trim_loss_pct <= 4.00: # Filter for naive pattern
                pattern.update({'trim_loss': trim_loss, "trim_loss_pct": trim_loss_pct, "count_cut":count_fg_cut})
                self.filtered_patterns.append(pattern)
        # Sort pattern
        self.filtered_patterns = sorted(self.filtered_patterns, key=lambda x: (x['trim_loss_pct'], -x['count_cut']))
        # print(f"filter pattern {len(self.filtered_patterns)}:{ self.filtered_patterns}")

        # Initiate dict
        self.chosen_stocks = {}

        # Filter stocks
        filtered_stocks = [self.filtered_patterns[i]['stock'] for i in range(len(self.filtered_patterns))]
        for stock_name, stock_info in self.start_stocks.items():
            if stock_name in filtered_stocks:
                self.chosen_stocks[stock_name]= {**stock_info}
    
    # PHASE 4: Optimize WEIGHT Patterns    
    def optimize_cut(self):
    
        # Create the solver instance (CBC Solver)
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Sets
        S = list(self.chosen_stocks.keys())
        P = list(range(len(self.filtered_patterns)))

        # PARAMETER - unit weight of stock
        c = {p: self.chosen_stocks[pattern["stock"]]["weight"] / self.chosen_stocks[pattern["stock"]]["width"] for p, pattern in enumerate(self.filtered_patterns)}

        # Decision variables (binary: choose pattern or not)
        x = {p: solver.IntVar(0, 1, f"X_{p}") for p in P}

        # Objective function: MINIMIZE total stock use
        objective = solver.Objective()
        for p in P:
            objective.SetCoefficient(x[p], 1)
        objective.SetMinimization()

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            # Lower bound constraint (demand for finished part)
            demand_weight_constraint = solver.Constraint(self.dual_finish[f]['need_cut'] - 0.019 * self.dual_finish[f]['average FC'], solver.infinity())
            for p in P:
                demand_weight_constraint.SetCoefficient(x[p], self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p])

            # Upper bound constraint
            upper_demand_weight_constraint = solver.Constraint(-solver.infinity(), self.dual_finish[f]['upper_bound'])
            for p in P:
                upper_demand_weight_constraint.SetCoefficient(x[p], self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p])

        # Solve the problem
        status = solver.Solve()

        self.solution_list = []
        if status == pywraplp.Solver.OPTIMAL:
            solution = [round(x[p].solution_value()) for p in P]

            # Extract results
            for i, pattern_info in enumerate(self.filtered_patterns):
                count = solution[i]
                if count > 0:
                    self.solution_list.append({"count": count, **pattern_info})

            self.probstt = "Solved"
        else:
            self.probstt = "Infeasible"

    def cbc_optimize_cut(self):
        # Sets
        S = list(self.chosen_stocks.keys())
        P = list(range(len(self.filtered_patterns)))

        # PARAMETER - unit weight of stock
        c = {p: self.chosen_stocks[pattern["stock"]]["weight"]/self.chosen_stocks[pattern["stock"]]["width"] for p, pattern in enumerate(self.filtered_patterns)}

        # Define VARIABLES
        x = {p: LpVariable(f"X_{p}",lowBound=0, upBound=1, cat='Integer') for p in P}
        
        # Create a LP minimization problem
        prob = LpProblem("PatternCuttingProblem", LpMinimize)
        
        # Objective function: MINIMIZE total stock use
        prob += lpSum(x[p] for p in P), "TotalStockUse"

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) >= self.dual_finish[f]['need_cut'], 
                f"DemandWeight{f}"
            )
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) <= self.dual_finish[f]['upper_bound'], 
                f"UpperDemandWeight{f}"
            )
        # Solve the problem GLPL - IBM solver
        print("solver CBC")
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs'], timeLimit=60))
        
        try: # Extract results
            solution = [
                        1 if (x[p].varValue > 0 and round(x[p].varValue) == 0) 
                        else round(x[p].varValue) 
                        for p in range(len(self.filtered_patterns))
                        ]
            self.solution_list = []
            for i, pattern_info in enumerate(self.filtered_patterns):
                count = solution[i]
                if count > 0:
                    self.solution_list.append({"count": count, **pattern_info})
            if sum(solution) > 0:
                self.probstt = "Solved"
            else: self.probstt = "Infeasible"
        except KeyError: self.probstt = "Infeasible" # khong co nghiem
    
    def _expand_solution_list(self):

        # List to hold the result
        self.expanded_solution_list = []

        # Process each item in the inventory
        for item in self.solution_list:
            count = item['count']
            # Replicate the entry 'count' times >1 to 'count' set to 1
            for _ in range(count):
                new_item = item.copy()  # Copy the item
                new_item['count'] = 1  # Set the count to 1
                self.expanded_solution_list.append(new_item)
    
    def find_final_solution_patterns(self):
        """ 
        Args:
            patterns [
                    {"stock":,'TP238H002948-1', 'inventory_id': 'TP238H002948-1',
                    'stock_weight': 4000, 'stock_width': 1219 'trim_loss': 48.0, 'trim_loss_pct': 3.938,
                    'explanation':, "cutting_date': , 'remark':,
                    'cuts': {'F200': 0, 'F198': 3, 'F197': 0, 'F196': 1, 'F190': 4, 'F511': 2, 'F203': 0}, 
                    'details:[]
                    }, 
                    {"stock":,'TP238H002948-2', ...
                    },
                ]
        """
        self._expand_solution_list()
        sorted_solution_list = sorted(self.expanded_solution_list, key=lambda x: (x['stock'],  x.get('trim_loss_pct', float('inf'))))
        self.overused_list = []
        take_stock = None
        for solution_pattern in sorted_solution_list:
            current_stock = solution_pattern['stock']
            if current_stock == take_stock:
                self.overused_list.append(solution_pattern)
            else:
                take_stock = current_stock
                self.final_solution_patterns.append(solution_pattern)
                
    def run(self, solver):
        #Phase 1
        self.create_finish_demand_by_line_w_naive_pattern()
        
        #Phase 2
        self.generate_patterns()

        #Phase 3
        self.filter_patterns_and_stocks_by_constr()
        
        #Phase 4
        if solver == "CBC":
            self.cbc_optimize_cut()
        else:
            self.optimize_cut()
        
        if self.probstt == 'Solved':
            self.find_final_solution_patterns()
            
# #HELPER
# import math

# def calculate_upper_bounds(finish, bound):
#     mean_3fc = {
#             f: (
#                 sum(v for v in (f_info['fc1'], f_info['fc2'], f_info['fc3']) if not math.isnan(v)) /
#                 sum(1 for v in (f_info['fc1'], f_info['fc2'], f_info['fc3']) if not math.isnan(v))
#             ) if any(not math.isnan(v) for v in (f_info['fc1'], f_info['fc2'], f_info['fc3'])) else float('nan')
#             for f, f_info in finish.items()
#         }
#     finish = {f: {**f_info, "mean_3fc": mean_3fc[f] if mean_3fc[f]>0 else -f_info['need_cut'] } for f, f_info in finish.items()}
        
#     # Need_cut van la so am
#     finish = {f: {**f_info, "upper_bound": -f_info['need_cut'] + f_info['mean_3fc']* bound} for f, f_info in finish.items()}
#     return finish      

# if __name__ == "__main__":
#     finish = {
#                   "F626": {
#                      "customer_name": "TEC",
#                      "width": 121.2,
#                      "need_cut": -7005.70698757764,
#                      "fc1": 16388.348,
#                      "fc2": 0.0,
#                      "fc3": 9008.282,
#                      "average FC": 8465.543333333333,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 300,
#                      "Max_weight": 500
#                   },
#                   "F636": {
#                      "customer_name": "TEC",
#                      "width": 196.0,
#                      "need_cut": -5663.586956521739,
#                      "fc1": 11943.8700096,
#                      "fc2": 174.68986079999996,
#                      "fc3": 6921.4064136,
#                      "average FC": 6346.655427999999,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 150,
#                      "Max_weight": 350
#                   },
#                   "F637": {
#                      "customer_name": "TEC",
#                      "width": 206.0,
#                      "need_cut": -5203.64751552795,
#                      "fc1": 10608.94386,
#                      "fc2": 0.0,
#                      "fc3": 6658.549940000001,
#                      "average FC": 5755.831266666667,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 300,
#                      "Max_weight": 500
#                   },
#                   "F624": {
#                      "customer_name": "TEC",
#                      "width": 110.0,
#                      "need_cut": -1280.4992236024846,
#                      "fc1": 1741.616,
#                      "fc2": 1048.29242,
#                      "fc3": 2165.562,
#                      "average FC": 1651.8234733333331,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 150,
#                      "Max_weight": 350
#                   },
#                   "F628": {
#                      "customer_name": "TEC",
#                      "width": 135.0,
#                      "need_cut": -203.65877329192546,
#                      "fc1": 2252.57308,
#                      "fc2": 1912.9525500000002,
#                      "fc3": 1331.6751600000002,
#                      "average FC": 1832.4002633333337,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 150,
#                      "Max_weight": 350
#                   },
#                   "F646": {
#                      "customer_name": "TEC",
#                      "width": 63.0,
#                      "need_cut": -4.0,
#                      "fc1": 390.9795,
#                      "fc2": 343.1219,
#                      "fc3": 588.3921,
#                      "average FC": 440.83116666666666,
#                      "1st Priority": "NQS",
#                      "2nd Priority": "NST",
#                      "3rd Priority": "HSC",
#                      "Min_weight": 100,
#                      "Max_weight": 300
#                   }
#                }
#     stocks = {
#             "HTV0348/24": {
#                "receiving_date": 44945,
#                "width": 1219,
#                "weight": 7809,
#                "warehouse": "NQS",
#                "status": "M:RAW MATERIAL",
#                "remark": "",
#                "min_margin":5
#             },
#             "HTV0349/24": {
#                "receiving_date": 44945,
#                "width": 1219,
#                "weight": 7904,
#                "warehouse": "NQS",
#                "status": "M:RAW MATERIAL",
#                "remark": "",
#                "min_margin":5
#             },
#             "HTV0340/24": {
#                "receiving_date": 44945,
#                "width": 1219,
#                "weight": 8363,
#                "warehouse": "NQS",
#                "status": "M:RAW MATERIAL",
#                "remark": "",
#                "min_margin":5
#             },
#             "HTV0341/24": {
#                "receiving_date": 44945,
#                "width": 1219,
#                "weight": 8948,
#                "warehouse": "NQS",
#                "status": "M:RAW MATERIAL",
#                "remark": "",
#                "min_margin":5
#             }
#          }
    
#     finish = calculate_upper_bounds(finish, 1.0)
#     prob = testDualProblem(finish, stocks)
#     prob.run()
    