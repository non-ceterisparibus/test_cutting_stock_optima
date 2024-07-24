import pandas as pd
import numpy as np
import copy
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

# DEFINE OBJECTS
class FinishObjects:
    # SET UP STOCKS AND FINISH
    def __init__(self, finish, PARAMS):
        self.upperbound = 2 # DEFAULT 2 months forecast
        self.spec = PARAMS['spec_name']
        self.thickness = PARAMS['thickness']
        self.maker = PARAMS['maker']
        self.type = PARAMS['type']
        self.finish =  finish

    def _calculate_upper_bounds(self): 
        # Need_cut van la so am
        if self.upperbound == 1:
            self.finish = {f: {**f_info, "upper_bound": -f_info['need_cut'] + f_info['fc1']} for f, f_info in self.finish.items()}
        elif self.upperbound == 2:
            self.finish = {f: {**f_info, "upper_bound": -f_info['need_cut'] + f_info['fc1'] + f_info['fc2']} for f, f_info in self.finish.items()}
        elif self.upperbound == 3:
            self.finish = {f: {**f_info, "upper_bound": -f_info['need_cut'] + f_info['fc1'] + f_info['fc2'] + f_info['fc3']} for f, f_info in self.finish.items()}
    
    def _reverse_need_cut_sign(self):
        for _, f_info in self.finish.items():
            if f_info['need_cut']  < 0:
                f_info['need_cut'] *= -1
            else: 
                f_info['need_cut'] = 0
                f_info['upper_bound'] += -f_info['need_cut']

    def update_bound(self,bound):
        # Default case
        if bound <= 3:
            self.upperbound = bound
            self._calculate_upper_bounds()
            self._reverse_need_cut_sign()
        else:
            raise ValueError("bound should be smaller than 3")
    
class StockObjects:
    # SET UP STOCKS AND FINISH
    def __init__(self,stocks, PARAMS):
        self.spec = PARAMS['spec_name']
        self.thickness = PARAMS['thickness']
        self.maker = PARAMS['maker']
        self.type = PARAMS['type']
        self.stocks = stocks
        
    def update_min_margin(self, margin_df):
        for s, s_info in self.stocks.items():
            if s_info['warehouse'] == "NQS":
                margin_filtered = margin_df[(margin_df['coil_center'] == "NQS") & (margin_df['Type'] == self.type)]
            else:
                margin_filtered = margin_df[(margin_df['coil_center'] == s_info['warehouse'])]

            min_trim_loss = self._find_min_trim_loss(margin_filtered)
            
            self.stocks[s] ={**s_info, "min_margin": min_trim_loss}

    def _find_min_trim_loss(self, margin_df):
        for _, row in margin_df.iterrows():
            thickness_range = row['Thickness']
            min_thickness, max_thickness = self._parse_thickness_range(thickness_range)
            if min_thickness < self.thickness <= max_thickness:
                return row['Min Trim loss (mm)']
        return None
        
    def _parse_thickness_range(self,thickness_str):
        if "≤" in thickness_str and "<" not in thickness_str:
            parts = thickness_str.split("≤")
            return (0, float(thickness_str.replace("≤", "")))
        elif "≤" in thickness_str and "T" in thickness_str:
            parts = thickness_str.split("≤")
            min_thickness = float(parts[0].replace("<T", "")) if parts[0] else float('-inf')
            max_thickness = float(parts[1]) if parts[1] else float('inf')
            return (min_thickness, max_thickness)
        elif ">" in thickness_str:
            parts = thickness_str.split(">")
            return (float(parts[1]), float('inf'))
        else:
            raise ValueError(f"Unsupported thickness range format: {thickness_str}")

# DEFINE PROBLEM
class DualProblem:
    def __init__(self, dual_finish, dual_stocks):
        self.len_stocks = len(dual_stocks)
        self.dual_finish = dual_finish
        self.dual_stocks = dual_stocks
        self.start_stocks = dual_stocks
        self.start_finish = dual_finish
        self.final_solution_patterns = []
        # [{'stock': 'TP238H002948-1', 'cuts': {'F200': 0, 'F198': 3, 'F197': 0, 'F196': 1, 'F190': 4, 'F511': 2, 'F203': 0}, 
        #  'trim_loss': 48.0, 'trim_loss_pct': 3.938}, 

    # PHASE 1: Naive/ Dual Pattern Generation
    def make_naive_patterns(self):
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
                    self.patterns.append({"stock": s, "cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct })

            if not feasible:
                pass
                # print(f"No feasible pattern was found for Stock {s} and FG {f}")

    def create_finish_demand_by_line_w_naive_pattern(self):
        """
        finish {finish: width, need_cut, upper_bound,fc1,fc2,fc3 } 
        Convert demand in KGs to demand in slice on naive pattern
        """
        dump_ls = {}
        for f, finish_info in self.dual_finish.items():
            try:
                non_zero_min = min([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns) if self.patterns[i]['cuts'][f] != 0])
            except ValueError:
                non_zero_min = 0
            dump_ls[f] = {**finish_info
                            ,"upper_demand_line": max([self.patterns[i]['cuts'][f] for i,_ in enumerate(self.patterns)])
                            ,"demand_line": non_zero_min }
       
        # Filtering the dictionary to include only items with keys in keys_to_keep
        self.dual_finish = {k: v for k, v in dump_ls.items() if v['upper_demand_line'] > 0} # xem lai dieu kien nay, tuc la neu cat dai nay voi stock hien co thì overcut lon
    
    # PHASE 2: Pattern Duality
    def _filter_out_overlap_stock(self):
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
        prob = LpProblem("NewPatternProblem", LpMaximize)

        # Decision variables - Pattern
        ap = {f: LpVariable(f"ap_{f}", 0, ap_upper_bound[f], cat="Integer") for f in self.dual_finish.keys()}

        # Objective function
        # maximize marginal_cut:
        prob += lpSum(ap[f] * demand_duals[f] for f in self.dual_finish.keys()), "MarginalCut"

        # Constraints - subject to stock_width - MIN MARGIN
        prob += lpSum(ap[f] * self.dual_finish[f]["width"] for f in self.dual_finish.keys()) <= width_s - MIN_MARGIN, "StockWidth_MinMargin"
        
        # Constraints - subject to trim loss 4% 
        prob += lpSum(ap[f] * self.dual_finish[f]["width"] for f in self.dual_finish.keys()) >= 0.96 * width_s , "StockWidth"

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

        marg_cost = value(prob.objective)
        pattern = {f: int(ap[f].varValue) for f in self.dual_finish.keys()}
        
        return marg_cost, pattern

    def _generate_dual_pattern(self):
        # Stock nao do toi uu hon stock khac o width thi new pattern luon bi chon cho stock do #FIX
        prob = LpProblem("GeneratePatternDual", LpMinimize)

        # Sets
        F = list(self.dual_finish.keys())
        P = list(range(len(self.patterns)))

        # Parameters
        s = {p: self.patterns[p]["stock"] for p in range(len(self.patterns))}
        a = {(f, p): self.patterns[p]["cuts"][f] for p in P for f in F}
        demand_finish = {f: self.dual_finish[f]["demand_line"] for f in F}
        upper_demand_finish = {f: self.dual_finish[f]["upper_demand_line"] for f in F}

        # Decision variables #relaxed integrality
        x = {p: LpVariable(f"x_{p}", 0, None, cat="Continuous") for p in P}

        # OBJECTIVE function minimize stock used:
        prob += lpSum(x[p] for p in P), "Cost"

        # Constraints
        for f in F:
            prob += lpSum(a[f, p] * x[p] for p in P) >= demand_finish[f], f"Demand_{f}"
            prob += lpSum(a[f, p] * x[p] for p in P) <= upper_demand_finish[f], f"UpperDemand_{f}" # ADD CONTRAINT UPPER

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

        # Extract dual values
        dual_values = {f: prob.constraints[f"Demand_{f}"].pi for f in F}

        ap_upper_bound = {f: max([self.patterns[i]['cuts'][f] for i,_ in enumerate(self.patterns)]) for f in self.dual_finish.keys()}
        demand_duals = {f: dual_values[f] for f in F}

        marginal_values = {}
        pattern = {}
        for s in self.dual_stocks.keys():
            marginal_values[s], pattern[s] = self._new_pattern_problem( #new pattern by line cut (trimloss), ignore weight
                self.dual_stocks[s]["width"], ap_upper_bound, demand_duals, self.dual_stocks[s]["min_margin"]
            )
            
        try:
            s = max(marginal_values, key=marginal_values.get) # pick the first stock if having same width
            new_pattern = {"stock": s, "cuts": pattern[s]}
        except ValueError:
            new_pattern = None
        return new_pattern
    
    # Solve Duality
    def generate_patterns(self):
        n = 0
        remove_stock = True
        self.max_key = None
        while remove_stock == True:
            self._filter_out_overlap_stock()
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
                if max_count > 1 and n < self.len_stocks - 2:
                    remove_stock = True
                    n +=1
                else: 
                    remove_stock = False

    # PHASE 3: Filter Patterns
    def filter_patterns_and_stocks_by_constr(self):
        # Initiate list
        self.filtered_patterns = []

        # Filter patterns
        for pattern in self.patterns:
            cuts_dict= pattern['cuts']
            width_s = self.start_stocks[pattern['stock']]['width']
            trim_loss = width_s - sum([self.start_finish[f]["width"] * cuts_dict[f] for f in cuts_dict.keys()])
            trim_loss_pct = round(trim_loss/width_s * 100, 3)
            if trim_loss_pct <= 4.00: # filter for naive pattern
                pattern.update({'trim_loss': trim_loss, "trim_loss_pct": trim_loss_pct})
                self.filtered_patterns.append(pattern)

        # Initiate dict
        self.chosen_stocks = {}

        # Filter stocks
        filtered_stocks = [self.filtered_patterns[i]['stock'] for i in range(len(self.filtered_patterns))]
        for stock_name, stock_info in self.start_stocks.items():
            if stock_name in filtered_stocks:
                self.chosen_stocks[stock_name]= {**stock_info}
    
    # PHASE 4: Optimize WEIGHT Patterns
    def optimize_cut(self):
        self.probstt = None
        # Parameters - unit weight
        c = {p: self.chosen_stocks[pattern['stock']]["weight"]/self.chosen_stocks[pattern['stock']]["width"] for p, pattern in enumerate(self.filtered_patterns)}

        # Create a LP minimization problem
        prob = LpProblem("PatternCuttingProblem", LpMinimize)

        # Define variables
        x = {p: LpVariable(f"x_{p}", 0, 1, cat='Integer') for p in range(len(self.filtered_patterns))} # tu tach ta stock dung nhieu lan thanh 2 3 dong

        # Objective function: minimize total stock use
        prob += lpSum(x[p]* c[p] for p in range(len(self.filtered_patterns))), "TotalStockUse"

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            prob += lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * x[p] * c[p] 
                          for p in range(len(self.filtered_patterns))) >= self.dual_finish[f]['need_cut'], f"DemandWeight{f}"
            prob += lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * x[p] * c[p] 
                          for p in range(len(self.filtered_patterns))) <= self.dual_finish[f]['upper_bound'], f"UpperDemandWeight{f}"
        
        # Solve the problem
        prob.solve()
        self.probstt = LpStatus[prob.status]

        if  self.probstt == "Optimal":
            # Extract results
            solution = [1 if (x[p].varValue > 0 and round(x[p].varValue)==0) else round(x[p].varValue) for p in range(len(self.filtered_patterns))]  # Fix integer
            self.solution_list = []
            for i, pattern_info in enumerate(self.filtered_patterns):
                count = solution[i]
                if count > 0:
                    self.solution_list.append({"count": count, **pattern_info})
        else: pass # khong co nghiem
    
    def find_final_solution_patterns(self):
        # Neu lap stock thi rm all pattern tru cai trim loss thap nhat va chay lai
        sorted_solution_list = sorted(self.solution_list, key=lambda x: (x['stock'],  x.get('trim_loss_pct', float('inf'))))
        self.overused_list = []
        take_stock = None
        for solution_pattern in sorted_solution_list:
            current_stock = solution_pattern['stock']
            if current_stock == take_stock:
                self.overused_list.append(solution_pattern)
            else:
                take_stock = current_stock
                self.final_solution_patterns.append(solution_pattern)

    def run(self):
        #Phase 1
        self.make_naive_patterns()
        self.create_finish_demand_by_line_w_naive_pattern()
        
        #Phase 2
        self.generate_patterns()

        #Phase 3
        self.filter_patterns_and_stocks_by_constr()
        
        #Phase 4
        self.optimize_cut()
        if self.probstt == 'Optimal':
            self.find_final_solution_patterns()

class LinearProblem:
    def __init__(self, finish, stocks):
        self.stocks = stocks
        self.finish = finish
        self.final_solution_patterns = []


class Cuttingtocks:
    def __init__(self, finish, stocks, PARAMS):
        self.S = StockObjects(stocks, PARAMS)
        self.F = FinishObjects(finish, PARAMS)
        self.over_cut = None

    def update(self, bound, margin_df):
        self.F.update_bound(bound)
        self.S.update_min_margin(margin_df)
    
    def set_dualprob(self):
        # Initiate problem 
        self.dualprob = DualProblem(self.F.finish, self.S.stocks)

    # Phase 5: Evaluation Over-cut / Stock Ratio
    def _count_weight(self):
        # Initialize an empty dictionary for the total sums
        total_sums = {}
        # Loop through each dictionary in the list
        for entry in self.dualprob.final_solution_patterns:
            count = entry['count']
            cuts = entry['cut_w']

            # Loop through each key in the cuts dictionary
            for key, value in cuts.items():
                # If the key is not already in the total_sums dictionary, initialize it to 0
                if key not in total_sums:
                    total_sums[key] = 0
                # Add the product of count and value to the corresponding key in the total_sums dictionary
                total_sums[key] += round(count * value,2)
        return total_sums

    def _calculate_finish_after_cut(self):
        # for all orginal finish, not only dual
        if self.dualprob.probstt == "Optimal":
            for i, sol in enumerate(self.dualprob.final_solution_patterns):
                s = self.dualprob.final_solution_patterns[i]['stock'] # stock cut
                cuts_dict = self.dualprob.final_solution_patterns[i]['cuts']
                weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * self.S.stocks[s]['weight']/self.S.stocks[s]['width'],3) for f in cuts_dict.keys()}
                self.dualprob.final_solution_patterns[i] = {**sol, "cut_w": weight_dict}
            # Total Cuts
            total_sums = self._count_weight()
            self.over_cut = {k: round(total_sums[k] - self.F.finish[k]['need_cut'],3) for k in total_sums.keys()} # can tinh overcut cho moi finish, du ko duoc cat trong list dual finish
        else: pass # ko co nghiem trong lan chay truoc do
        
    def solve_dualprob(self):
        # Run and calculate results
        self.dualprob.run()
        self._calculate_finish_after_cut()
        
        return self.dualprob.probstt, self.dualprob.final_solution_patterns, self.over_cut
        
    def _check_remain_stocks(self):
        # Extract stocks from final_solution_patterns
        taken_stocks = {p['stock'] for p in self.dualprob.final_solution_patterns} 
        # Update need cut
        for f, f_info in self.dualprob.dual_finish.items():
            if self.over_cut[f] < 0:
                f_info['need_cut'] = -self.over_cut[f]
            else: 
                f_info['need_cut'] = 0
                f_info['upper_bound'] += -self.over_cut[f]
        # Find remained_stocks dictionary
        self.remained_stocks = {
            s: {**s_info}
            for s, s_info in self.dualprob.dual_stocks.items()
            if s not in taken_stocks
        }
        
    def check_status(self):
        self._check_remain_stocks()
        # CHECK FOR ANOTHER ROUND
        if not self.dualprob.overused_list or not self.remained_stocks:
            print("FINISH CUTTING")
            return False
        else:
            # go back
            print("CONTINUE CUTTING")
            return True
        
    def refresh_data(self):
        self.dualprob.dual_stocks = copy.deepcopy(self.remained_stocks)
    
# if __name__ == "__main__":
#     # LOAD CONFIG & DATA
#     PARAMS = {"spec_name": "JSH590R-PO" # yeu cau chuan hoa du lieu OP - PO
#                 ,"thickness": 2
#                 ,"maker" : "CSC"
#                 ,"type": "Carbon"
#                 ,'code': 'POSCOVN JSH590R-PO 2.0'
#                 }

#     margin_df = pd.read_csv('data/min_margin.csv')
#     spec_type = pd.read_csv('data/spec_type.csv')
#     coil_priority = pd.read_csv('data/coil_data.csv')

#     # CONVERT F-S TO DICT
#     stocks = {'TP235H002656-2': {'receiving_date': 45150, 'width': 1219, 'weight': 4332.5,"warehouse": "HSC"}, 
#             'TP235H002656-1': {'receiving_date': 44951, 'width': 1219, 'weight': 4332.5,"warehouse": "HSC"},
#             'TP232H001074': {'receiving_date': 45040, 'width': 1233, 'weight': 7105.0,"warehouse": "HSC"},
#             'TP232H001073': {'receiving_date': 45153, 'width': 1233, 'weight': 7550.0,"warehouse": "HSC"}, 
#             'TP236H005198': {'receiving_date': 45011, 'width': 1136, 'weight': 8000.0,"warehouse": "HSC"},
#             'TP235H002652': {'receiving_date': 45039, 'width': 1219, 'weight': 8400.0,"warehouse": "HSC"}, 
#             'TP235H002654': {'receiving_date': 45045, 'width': 1219, 'weight': 8500.0,"warehouse": "HSC"}, 
#             'TP232H001072': {'receiving_date': 45013, 'width': 1233, 'weight': 8675.0,"warehouse": "HSC"}, 
#             'TP235H002655': {'receiving_date': 45229, 'width': 1219, 'weight': 8845.0,"warehouse": "HSC"}, 
#             'TP235H002653': {'receiving_date': 45045, 'width': 1219, 'weight': 8855.0,"warehouse": "HSC"}, 
#             'TP232H001075': {'receiving_date': 45247, 'width': 1233, 'weight': 9630.0,"warehouse": "HSC"}
#             }

#     finish = {'F23': {'width': 306.0,  'need_cut': 839.0,  'upper_bound': 1548.5841833599998,  'fc1': 2365.2806112,  'fc2': 3692.5657704,  'fc3': 3457.8613836}, 
#             'F22': {'width': 205.0,  'need_cut': 498.7908121410992,  'upper_bound': 3362.2258921410994,  'fc1': 9544.7836,  'fc2': 5494.6232,  'fc3': 3908.5464},
#             'F21': {'width': 188.0,  'need_cut': 30772.599709771595,  'upper_bound': 39966.4243228516,  'fc1': 30646.0820436,  'fc2': 35762.3146452,  'fc3': 34039.2591132},
#             'F20': {'width': 175.0,  'need_cut': 28574.78588807786,  'upper_bound': 36618.447115077855,  'fc1': 26812.20409,  'fc2': 31288.38713,  'fc3': 29780.88883}, 
#             'F19': {'width': 155.0,  'need_cut': 4401.8405357987585,  'upper_bound': 5851.570175548759,  'fc1': 4832.4321325,  'fc2': 5639.1860525,  'fc3': 5367.4857775},
#             'F18': {'width': 133.0,  'need_cut': 400.0,  'upper_bound': 795.8254562499999,  'fc1': 1319.4181875,  'fc2': 759.546375,  'fc3': 540.295875}, 
#             'F17': {'width': 120.0,  'need_cut': 1751.0,  'upper_bound': 2526.6533504,  'fc1': 2585.511168,  'fc2': 4319.793456,  'fc3': 3797.778504}, 
#             'F24': {'width': 82.0,  'need_cut': 977.9362646180011,  'upper_bound': 1585.531389098001,  'fc1': 2025.3170816,  'fc2': 3383.8382072,  'fc3': 2974.9264948}
#     }

#     # SETUP
#     steel = Cuttingtocks(finish, stocks, PARAMS)
#     steel.update(bound = 2, margin_df=margin_df)
#     final_solution_patterns, over_cut = steel.solve_dualprob()
#     print(f'Take stock {[p['stock'] for p in final_solution_patterns]}')
#     print(f'overcut amount {over_cut}')
#     print(over_cut)
