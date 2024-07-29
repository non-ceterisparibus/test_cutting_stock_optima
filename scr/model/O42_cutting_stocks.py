import pandas as pd
import numpy as np
import copy
from model.O31_steel_objects import FinishObjects, StockObjects
from model.O41_dual_solver import DualProblem
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','cuts',"cut_w"]}
        for cut, line in item['cuts'].items():
            if line > 0:
                flattened_item = {**common_data, 'cuts': cut, 'lines': line}
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

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
#                 ,'code': 'CSC JSH590R-PO 2.0'
#                 }

#     margin_df = pd.read_csv('scr/data/min_margin.csv')
#     spec_type = pd.read_csv('scr/data/spec_type.csv')
#     # coil_priority = pd.read_csv('data/coil_data.csv')

#     # CONVERT F-S TO DICT
#     stocks = {
#             "TP232H001075": {"receiving_date": 45175, "width": 1233,"weight": 9630,"warehouse": "HSC"},
#             "TP235H002653": {"receiving_date": 45278, "width": 1219,"weight": 8855,"warehouse": "HSC"},
#             "TP235H002655": {"receiving_date": 45046,"width": 1219,"weight": 8845,"warehouse": "HSC"},
#             "TP232H001072": {"receiving_date": 45080,"width": 1233,"weight": 8675,"warehouse": "HSC"},
#             "TP235H002654": {"receiving_date": 45172,"width": 1219,"weight": 8500,"warehouse": "HSC"},
#             "TP235H002652": {"receiving_date": 45278,"width": 1219,"weight": 8400,"warehouse": "HSC"},
#             "TP236H005198": {"receiving_date": 45016,"width": 1136,"weight": 8000,"warehouse": "HSC"},
#             "TP232H001073": {"receiving_date": 45031,"width": 1233,"weight": 7550,"warehouse": "HSC"},
#             "TP232H001074": {"receiving_date": 45137,"width": 1233,"weight": 7105,"warehouse": "HSC"},
#             "TP235H002656-2": {   "receiving_date": 45142,   "width": 1219,   "weight": 5000,   "warehouse": "HSC"},
#             "TP235H002656-1": {   "receiving_date": 45017,   "width": 1219,   "weight": 4333,   "warehouse": "HSC"}
#          }

#     finish ={"F33": {"customer_name": "CIC", "width": 188.0, "need_cut": -30772.599709771595, "fc1": 30646.0820436, "fc2": 35762.3146452, "fc3": 34039.2591132},
#             "F32": {"customer_name": "CIC","width": 175.0,"need_cut": -28574.78588807786,"fc1": 26812.20409,"fc2": 31288.38713,"fc3": 29780.88883},
#             "F31": {"customer_name": "CIC","width": 155.0,"need_cut": -4401.8405357987585,"fc1": 4832.4321325,"fc2": 5639.1860525,"fc3": 5367.4857775},
#             "F29": {"customer_name": "CIC","width": 120.0,"need_cut": -1751.0,"fc1": 2585.511168,"fc2": 4319.793456,"fc3": 3797.778504},
#             "F37": {"customer_name": "CIC","width": 82.0,"need_cut": -977.9362646180011,"fc1": 2025.3170816,"fc2": 3383.8382072,"fc3": 2974.9264948},
#             "F36": {"customer_name": "CIC","width": 306.0,"need_cut": -839.0,"fc1": 2365.2806112,"fc2": 3692.5657704,"fc3": 3457.8613836},
#             "F34": {"customer_name": "CIC","width": 205.0,"need_cut": -498.7908121410992,"fc1": 9544.7836,"fc2": 5494.6232,"fc3": 3908.5464},
#             "F30": {"customer_name": "CIC","width": 133.0,"need_cut": -400.0,"fc1": 1319.4181875,"fc2": 759.546375,"fc3": 540.295875},
#             }

#     # SETUP
#     steel = Cuttingtocks(finish, stocks, PARAMS)
#     steel.update(bound = 2, margin_df=margin_df)
#     steel.set_dualprob()
#     stt, final_solution_patterns, over_cut = steel.solve_dualprob()
#     print(f'Take stock {[p['stock'] for p in final_solution_patterns]}')
#     print(f'overcut amount {over_cut}')
#     print(over_cut)
#     if not final_solution_patterns:
#         print(f"The solution - bound {2} is empty.")

#     else:
#         df = transform_to_df(final_solution_patterns)
#         filename = f"scr/results/solution-.xlsx"
#         df.to_excel(filename, index=False)

    
