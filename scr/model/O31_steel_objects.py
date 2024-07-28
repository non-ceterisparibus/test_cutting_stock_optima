import pandas as pd
import numpy as np
import json
import datetime
import copy

# INPUT
margin_df = pd.read_csv('scr/data/min_margin.csv')
spec_type = pd.read_csv('scr/data/spec_type.csv')
# coil_priority = pd.read_csv('data/coil_data.csv')

# DEFINE OBJECTS
class FinishObjects:
    # SET UP FINISH
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
    # SET UP STOCKS
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

