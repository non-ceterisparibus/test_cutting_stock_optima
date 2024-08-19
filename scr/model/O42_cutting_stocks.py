import numpy as np
import pandas as pd
import copy
import statistics
from model.O31_steel_objects import FinishObjects, StockObjects
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value

# DEFINE PROBLEM
class DualProblem:
    def __init__(self, dual_finish, dual_stocks):
        self.len_stocks = len(dual_stocks)
        self.dual_finish = dual_finish
        self.dual_stocks = dual_stocks
        self.start_stocks = dual_stocks
        self.start_finish = dual_finish
        self.final_solution_patterns = []

    # PHASE 1: Naive/ Dual Pattern Generation
    def _make_naive_patterns(self):
        """
        Generates patterns of feasible cuts from stock width to meet specified finish widths.
        patterns [{'inventory_id': 'TP238H002948-1', 'stock_weight': 
                    'trim_loss': 48.0, 'trim_loss_pct': 3.938}
                  'cuts': {'F200': 0, 'F198': 3, 'F197': 0, 'F196': 1, 'F190': 4, 'F511': 2, 'F203': 0}, 
                , ]
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
                                          'trim_loss_mm':trim_loss, "trim_loss_pct": trim_loss_pct ,
                                          "explanation":"",'remark':"","cutting_date":"",
                                          "stock_weight": self.dual_stocks[s]['weight'], 'stock_width':self.dual_stocks[s]['width'],
                                          "cuts": cuts_dict,
                                          "details": [{'order_no': f, 'width':self.dual_finish[f]['width'], 'lines': cuts_dict[f]} for f in cuts_dict.keys()] 
                    })

            if not feasible:
                pass
                # print(f"No feasible pattern was found for Stock {s} and FG {f}")

    def create_finish_demand_by_line_w_naive_pattern(self):
        self._make_naive_patterns()
        # print(len(self.patterns))
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
            cuts_dict =pattern[s]
            new_pattern = {"stock":s, "inventory_id": s,"explanation":"",'remark':"","cutting_date":"",
                           'stock_weight': self.dual_stocks[s]['weight'], 
                           'stock_width': self.dual_stocks[s]["width"],
                           "cuts": pattern[s],
                           "details": [{'order_no': f, 
                                        'width':self.dual_finish[f]['width'], 
                                        'lines': cuts_dict[f]} 
                                       for f in cuts_dict.keys()] 
                           }
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
                pattern.update({'trim_loss_mm': trim_loss, "trim_loss_pct": trim_loss_pct})
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

        # Parameters - unit weight
        c = {p: self.chosen_stocks[pattern['stock']]["weight"]/self.chosen_stocks[pattern['stock']]["width"] for p, pattern in enumerate(self.filtered_patterns)}

        # Create a LP minimization problem
        prob = LpProblem("PatternCuttingProblem", LpMinimize)

        # Define variables
        x = {p: LpVariable(f"x_{p}", 0, 1, cat='Integer') for p in range(len(self.filtered_patterns))} # tu tach ta stock dung nhieu lan thanh 2 3 dong

        # Objective function: minimize total stock use
        prob += lpSum(x[p] for p in range(len(self.filtered_patterns))), "TotalStockUse"

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            prob += lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * x[p] * c[p] 
                          for p in range(len(self.filtered_patterns))) >= self.dual_finish[f]['need_cut'], f"DemandWeight{f}"
            prob += lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * x[p] * c[p] 
                          for p in range(len(self.filtered_patterns))) <= self.dual_finish[f]['upper_bound'], f"UpperDemandWeight{f}"
        
        # Solve the problem
        prob.solve()

        try:
            # Extract results
            solution = [1 if (x[p].varValue > 0 and round(x[p].varValue)==0) else round(x[p].varValue) for p in range(len(self.filtered_patterns))]  # Fix integer
            self.solution_list = []
            for i, pattern_info in enumerate(self.filtered_patterns):
                count = solution[i]
                if count > 0:
                    self.solution_list.append({"count": count, **pattern_info})
            self.probstt = "Solved" #may not Optimal as pulp defined
        except KeyError: self.probstt = "Infeasible" # khong co nghiem
    
    def find_final_solution_patterns(self):
        """ 
        patterns [
        {"stock":,'TP238H002948-1', 'inventory_id': 'TP238H002948-1',
        'stock_weight': 4000, 'stock_width': 1219 'trim_loss_mm': 48.0, 'trim_loss_pct': 3.938,
        'explanation':, "cutting_date': , 'remark':,
        'cuts': {'F200': 0, 'F198': 3, 'F197': 0, 'F196': 1, 'F190': 4, 'F511': 2, 'F203': 0}, 
        'details:[]
        }, 
        """
        
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
        self.create_finish_demand_by_line_w_naive_pattern()
        
        #Phase 2
        self.generate_patterns()

        #Phase 3
        self.filter_patterns_and_stocks_by_constr()
        
        #Phase 4
        self.optimize_cut()
        if self.probstt == 'Solved':
            self.find_final_solution_patterns()

class RewindProb(DualProblem):

  def __init__(self, finish, stock):
    super().__init__(finish, {} ) # tao dual_stocks va start stocks sau
    self.ratio = 0.5 # default ratio --> NEED TO CONSIDER SMALLEST LEFT OVER ALLOWED
    self.stock_key = list(stock.keys())[0]
    self.og_weight = stock[self.stock_key]['weight']
    self.stock = stock
    
  def _rewind_ratio(self):
    # xac dinh ratio da phai tinh den weight cat
    coil_weight = [round(self.dual_finish[f]["need_cut"] * self.stock[self.stock_key]['width'] /self.dual_finish[f]["width"], 3) for f in self.dual_finish.keys()]
    self.med_demand_weight= statistics.median(coil_weight) # cho phep cat du 1 chut
    
  def _check_rewind_coil(self):
    #remained rewind stock weight should be in this range 
    min_coil_weight = [round(self.dual_finish[f]["Min_weight"] * self.stock[self.stock_key]['width'] /self.dual_finish[f]["width"],3) for f in self.dual_finish.keys()]
    min_w = statistics.median(min_coil_weight)
    return min_w
  
  def create_new_stocks_set(self ):
    self._rewind_ratio()
    min_w = self._check_rewind_coil()
    print(f"start_stocks: {self.start_stocks}")
    if self.med_demand_weight> 0 and min_w < self.og_weight - self.med_demand_weight:
        i = 1
        while i < 3:
            self.dual_stocks[f'{self.stock_key}-Re{i}'] = self.stock[self.stock_key].copy()
            if i == 1: 
                self.dual_stocks[f'{self.stock_key}-Re{i}']['weight'] = self.med_demand_weight
                print(f"cut rewind weight {self.med_demand_weight}")
            else: 
                self.dual_stocks[f'{self.stock_key}-Re{i}'].update({'weight': float(self.og_weight) - float(self.med_demand_weight),
                                                                    'status': "R:REWIND"}) # we have new set of stock
                # print(f"remained weight {self.og_weight - self.med_demand_weight}")
            i += 1   
        self.start_stocks = copy.deepcopy(self.dual_stocks) 
        print(self.start_stocks)
        
    else: print(f" rewind_coil too small {self.og_weight - self.med_demand_weight}")

class CuttingStocks:
    def __init__(self, finish, stocks, PARAMS):
        """
        -- Input --
        PARAMS: {
            "spec_name": "JSH270C-PO",
            "type": "Carbon",
            "thickness": 3.0,
            "maker": "CSVC",
            "code": "CSVC JSH270C-PO 3.0"
        stocks = {"HTV0269/24-D1": {   "receiving_date": 44961,   "width": 1158, 
                                        "weight": 3936,   "warehouse": "NQS"}
        finish = {"F288": {   "customer_name": "VPIC1",   "width": 143.5,  "need_cut": -1300.0,  
                            "fc1": 2724.48,   "fc2": 1776.9600000000003,   "fc3": 1752.9600000000003,  
                            "1st Priority": "HSC",   "2nd Priority": "x",   "3rd Priority": "x",  
                            "Min_weight": 0.0,   "Max_weight": 0.0},
                  "F290": {...}
                  }
        """
        self.S = StockObjects(stocks, PARAMS)
        self.F = FinishObjects(finish, PARAMS)
        self.over_cut = None

    def update(self, bound, margin_df):
        self.F.update_bound(bound)
        self.S.update_min_margin(margin_df)
     
    def _stock_weight_threshold_by_width(self, min_w, max_w):
        """_Create a diction of min and weight of coil that will produce same div of FG_

        Args:
            min_w (_int_): Min Weight of Cut Coil by Customer x FG Codes
            max_w (_int_): Max Weight of Cut Coil by Customer x FG Codes
        Result:
        {1080: {'min': 1588.235294117647,
                'max': [2842.1052631578946,
                        5684.210526315789,
                        8526.315789473683,
                        11368.421052631578]},
        1219: {'min': 1792.6470588235295,
               'max': [3207.8947368421054,
                        6415.789473684211,
                        9623.684210526317,
                        12831.578947368422]}}
        """
        
        self.stock_width =list(set([sv['width'] for _ ,sv in self.S.stocks.items()]))
        max_f_width = max([fv['width'] for _, fv in self.F.finish.items()])
        min_f_width = min([fv['width'] for _, fv in self.F.finish.items()])
        self.mm_stock_weight = {}
        for c_width in self.stock_width:
            # find weight range with min-max weight
            # min
            if min_w == 0.0:
                min_coil_weight = 0
            else:
                min_coil_weight = min_w * c_width/min_f_width
            # max
            if max_w == 0.0:
                max_coil_weight = float('inf')
            else:
                max_coil_weight = max_w * c_width/ max_f_width
            
            # check if min_coil_weight << max_coil_weight

            max_threshold = []
            for i in range(4):
                max_t = max_coil_weight * (i+1)
                max_threshold.append(max_t)
            self.mm_stock_weight[c_width] = {"min":min_coil_weight, "max": max_threshold}

    def _filter_min_stock(self):
        "Remove MC that too small, 1 line cut [weight] < Min Weight "
        self.filtered_stocks ={}
        for s_width, sv in self.mm_stock_weight.items():
            min_w = sv['min']
            filtered_min_stocks = {k: v for k, v in self.S.stocks.items() if (v['width'] == s_width and v['weight'] >= min_w)}
            self.filtered_stocks.update(filtered_min_stocks)
        
    def filter_stocks(self, min_weight = None,max_weight = None):
        if min_weight == None  and max_weight == None:
            self.filtered_stocks = copy.deepcopy(self.S.stocks)
        else:
            self._stock_weight_threshold_by_width(min_weight, max_weight)
            self._filter_min_stock()

    def set_prob(self, prob_type):
        if len(self.filtered_stocks) > 0 and prob_type == "Dual":
            # Initiate problem 
            self.prob = DualProblem(self.F.finish, self.filtered_stocks)
            return True # continue to run solve prob
        elif len(self.filtered_stocks) > 0 : 
            self.prob = RewindProb(self.F.finish, self.filtered_stocks)
            self.prob.create_new_stocks_set()
            return True
        else: #ko co stock de cat
            return False

    # Phase 5: Evaluation Over-cut / Stock Ratio
    def _count_weight(self):
        # Initialize an empty dictionary for the total sums
        total_sums = {}
        # Loop through each dictionary in the list
        for entry in self.prob.final_solution_patterns:
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

    def _calculate_div_ratio(self,s):
        """
        {1080: {'min': 1588.235294117647,
                'max': [2842.1052631578946,
                        5684.210526315789,
                        8526.315789473683,
                        11368.421052631578]},
        1219: {'min': 1792.6470588235295,
               'max': [3207.8947368421054, 3 2
                        6415.789473684211, 2 3
                        9623.684210526317, 1 4
                        12831.578947368422]}} 0 5
        """
        # doi chieu stock weight voi range max in mm_stock_weight by stock_width
        stocks_width = self.prob.start_stocks[s]['width']
        to_verify_range = sorted(self.mm_stock_weight[stocks_width]['max'], reverse=True)
        self.div_ratio = 0
        for i, w in enumerate(to_verify_range):
            if self.prob.start_stocks[s]['weight'] >= w:
                self.div_ratio =  (3-i) + 2
                break
    
    def _calculate_finish_after_cut_by_mm_weight(self):
        # for all orginal finish, not only dual
        if self.prob.probstt == "Solved":
            for i, sol in enumerate(self.prob.final_solution_patterns):
                s = self.prob.final_solution_patterns[i]['stock'] # stock cut
                self._calculate_div_ratio(s)
                cuts_dict = self.prob.final_solution_patterns[i]['cuts']
                weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * self.prob.start_stocks[s]['weight']/self.prob.start_stocks[s]['width'],3) for f in cuts_dict.keys()}
                if self.div_ratio == 0: 
                    rmark_note = "" 
                else: rmark_note = f"chat {self.div_ratio} phan"
                self.prob.final_solution_patterns[i] = {**sol, 
                                                        "cut_w": weight_dict,
                                                        }
                self.prob.final_solution_patterns[i].update({"remark": rmark_note})
            # Total Cuts
            total_sums = self._count_weight()
            self.over_cut = {k: round(total_sums[k] - self.F.finish[k]['need_cut'],3) for k in total_sums.keys()} # can tinh overcut cho moi finish, du ko duoc cat trong list dual finish
        else: pass # ko co nghiem trong lan chay truoc do
    
    def _calculate_finish_after_cut(self):
        # for all orginal finish, not only dual
        if self.prob.probstt == "Solved":
            for i, sol in enumerate(self.prob.final_solution_patterns):
                s = self.prob.final_solution_patterns[i]['stock'] # stock cut
                # self._calculate_div_ratio(s)
                cuts_dict = self.prob.final_solution_patterns[i]['cuts']
                weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * self.prob.start_stocks[s]['weight']/self.prob.start_stocks[s]['width'],3) for f in cuts_dict.keys()}

                self.prob.final_solution_patterns[i] = {**sol, 
                                                        "cut_w": weight_dict,
                                                        }
            # Total Cuts
            total_sums = self._count_weight()
            self.over_cut = {k: round(total_sums[k] - self.F.finish[k]['need_cut'],3) for k in total_sums.keys()} # can tinh overcut cho moi finish, du ko duoc cat trong list dual finish
        else: pass # ko co nghiem trong lan chay truoc do
    
    def solve_prob(self):
        # Run and calculate results
        self.prob.run()
        self._calculate_finish_after_cut()
        
        return self.prob.probstt, self.prob.final_solution_patterns, self.over_cut

    def _check_remain_stocks(self):
        # Extract stocks from final_solution_patterns
        taken_stocks = {p['stock'] for p in self.prob.final_solution_patterns} 
        # Update need cut
        for f, f_info in self.prob.dual_finish.items():
            if self.over_cut[f] < 0:
                f_info['need_cut'] = -self.over_cut[f]
            else: 
                f_info['need_cut'] = 0
                f_info['upper_bound'] += -self.over_cut[f]
                
        # Find remained_stocks dictionary
        self.remained_stocks = {
            s: {**s_info}
            for s, s_info in self.prob.dual_stocks.items()
            if s not in taken_stocks
        }
        
    def check_status(self):
        self._check_remain_stocks()
        # CHECK FOR ANOTHER ROUND
        if not self.prob.overused_list or not self.remained_stocks:
            print("FINISH CUTTING")
            return False
        else:
            # go back
            print("CONTINUE CUTTING")
            return True
        
    def refresh_data(self):
        self.prob.dual_stocks = copy.deepcopy(self.remained_stocks)
