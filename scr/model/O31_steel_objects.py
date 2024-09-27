import pandas as pd
import numpy as np
import math
import os

# INPUT & CONFIG
margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')
 
global max_bound
max_bound = float(os.getenv('MAX_BOUND', '5.0'))

# DEFINE OBJECTS
class FinishObjects:
    """
    SET UP FINISH
    finish {"customer_name","width", "need_cut", 
    "fc1", "fc2", "fc3",
    "1st Priority", "2nd Priority", "3rd Priority",
    "Min_weight", "Max_weight"}
    """
    def __init__(self, finish, PARAMS):
        self.upperbound = 2 # DEFAULT 2 months forecast
        self.spec = PARAMS['spec_name']
        self.thickness = PARAMS['thickness']
        self.maker = PARAMS['maker']
        self.type = PARAMS['type']
        self.finish =  finish
        
    def reverse_need_cut_sign(self):
        for _, f_info in self.finish.items():
            if f_info['need_cut']  < 0:
                f_info['need_cut'] *= -1
            else: 
                f_info['need_cut'] = 0
                
    def _calculate_upper_bounds(self,bound):
        average_fc = {
            f: (
                sum(v for v in (f_info['fc1'], f_info['fc2'], f_info['fc3']) if not math.isnan(v)) /
                sum(1 for v in (f_info['fc1'], f_info['fc2'], f_info['fc3']) if not math.isnan(v))
            ) if any(not math.isnan(v) for v in (f_info['fc1'], f_info['fc2'], f_info['fc3'])) else float('nan')
            for f, f_info in self.finish.items()
        }
        self.finish = {f: {**f_info, "average FC": average_fc[f] if average_fc[f] > 0 else f_info['need_cut'] } for f, f_info in self.finish.items()}
        
        # Need_cut doi thanh so duong
        self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['average FC']* bound} for f, f_info in self.finish.items()}
      
    def update_bound(self,bound):
        # Need_cut doi thanh so duong
        if bound <= max_bound:
            self.upperbound = bound
            self._calculate_upper_bounds(bound)
        else:
            raise ValueError(f"bound should be smaller than {max_bound}")
    
class StockObjects:
    """
    SET UP STOCKS
    Stock: {receiving_date, width, weight, status, remark}
    """
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

