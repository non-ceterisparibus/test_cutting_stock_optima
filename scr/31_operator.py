import pandas as pd
import numpy as np
import json
import datetime
import copy

# INPUT
margin_df = pd.read_csv('data/min_margin.csv')
spec_type = pd.read_csv('data/spec_type.csv')
coil_priority = pd.read_csv('data/coil_data.csv')

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
        # FIX BY THE OPERATOR AND THE BOUND calculate upper_bound according to the (remained) need_cut and
        if self.upperbound == 1:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1']} for f, f_info in self.finish.items()}
        elif self.upperbound == 2:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1'] + f_info['fc2']} for f, f_info in self.finish.items()}
        elif self.upperbound == 3:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1'] + f_info['fc2'] + f_info['fc3']} for f, f_info in self.finish.items()}

    def update_bound(self,bound):
        if bound <= 3:
            self.upperbound = bound
            self._calculate_upper_bounds()
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

# SETUP
with open('job_list.json', 'r') as file:
    job_list = json.load(file)

with open('finish_list_.json', 'r') as file:
    finish_list = json.load(file)

with open('stocks_list.json', 'r') as file:
    stocks_list = json.load(file)

# PROCESS
for i, job in enumerate(job_list['jobs']):
    param = job['param']
    current_stocks = stocks_list['params'][i]
    PARAM = current_stocks[param]
    print(PARAM)
    print('\n next \n')
    stocks = current_stocks['stocks']
    current_finish = finish_list['params'][i]

    for cust_fin in current_finish['customer']:
        customer = list(cust_fin.keys())[0]
        finish = cust_fin[customer]
        # print('\n next \n')

        # then start process each finish set and stock set

        # before get new finish set, count how many stocks left
        # refresh stocks


