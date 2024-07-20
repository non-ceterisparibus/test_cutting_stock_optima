### LOAD FINISH AND MOTHER COIL BY PARAMS - CUSTOMER
import pandas as pd
import numpy as np

# INPUT

class SteelCutting:
    # SET UP STOCKS AND FINISH
    def __init__(self):
        self.finish_key = 'order_id'
        self.finhish_columns = ["width", "need_cut", "fc1", "fc2", "fc3"]
        self.stock_key = "inventory_id"
        self.stock_columns = ['receiving_date',"width", "weight",'warehouse']
        self.upperbound = 2 # DEFAULT 2 months forecast
    
    def setup(self, finish_df, stock_df, PARAMS):
        self.finish_df = finish_df # df by customer
        self.stock_df = stock_df
        self.filter_finish_df_to_dict()
        self.filter_multi_stock_df_to_dict()
        self.calculate_upper_bounds()
        self.spec = PARAMS['spec_name']
        self.thickness = PARAMS['thickness']
        self.maker = PARAMS['maker']
        self.type = PARAMS['type']


    def filter_finish_df_to_dict(self):

        sorted_df = self.finish_df.sort_values(by=['width','need_cut'], ascending=[False,False])
        sorted_df['need_cut'] = sorted_df['need_cut'] * -1

        # Initialize result dictionary - take time if the list long
        self.finish = {f"F{int(row[self.finish_key])}": {column: row[column] for 
                                              column in self.finhish_columns} for 
                                              _, row in sorted_df.iterrows()}
    
    def filter_multi_stock_df_to_dict(self):
        # Sort data according to the priority of FIFO
        sorted_df = self.stock_df.sort_values(by=['weight','receiving_date'], ascending=[True, False])
  
        # Set the index of the DataFrame to 'stock_id'
        sorted_df.set_index(self.stock_key, inplace=True)

        # Convert DataFrame to dictionary
        self.stocks = sorted_df[self.stock_columns].to_dict(orient='index')

    def update_min_margin(self, margin_df):
        for s, s_info in self.stocks.items():
            if s_info['warehouse'] == "NQS":
                margin_filtered = margin_df[(margin_df['coil_center'] == "NQS") & (margin_df['Type'] == self.type)]
            else:
                margin_filtered = margin_df[(margin_df['coil_center'] == s_info['warehouse'])]

            min_trim_loss = self.find_min_trim_loss(margin_filtered)
            
            self.stocks[s] ={**s_info, "min_margin": min_trim_loss}

    
    def calculate_upper_bounds(self): 
        # FIX BY THE OPERATOR AND THE BOUND calculate upper_bound according to the (remained) need_cut and
        if self.upperbound == 1:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1']} for f, f_info in self.finish.items()}
        elif self.upperbound == 2:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1'] + f_info['fc2']} for f, f_info in self.finish.items()}
        elif self.upperbound == 3:
            self.finish = {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1'] + f_info['fc2'] + f_info['fc3']} for f, f_info in self.finish.items()}

    def find_min_trim_loss(self, margin_df):
        for _, row in margin_df.iterrows():
            thickness_range = row['Thickness']
            min_thickness, max_thickness = self._parse_thickness_range(thickness_range)
            if min_thickness < self.thickness <= max_thickness:
                return row['Min Trim loss (mm)']
        return None

    def update_bound(self,bound):
        if bound <= 3:
            self.upperbound = bound
            self.calculate_upper_bounds()
        else:
            raise ValueError("bound should be smaller than 3")
        
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
    
    def make_naive_patterns(self):
        """
        Generates patterns of feasible cuts from stock width to meet specified finish widths.
        """
        self.patterns = []
        for f in self.finish:
            feasible = False
            for s in self.stocks:
                # max number of f that fit on s
                num_cuts_by_width = int((self.stocks[s]["width"]-self.stocks[s]["min_margin"]) / self.finish[f]["width"])
                # max number of f that satisfied the need cut WEIGHT BOUND
                num_cuts_by_weight = int((self.finish[f]["upper_bound"] * self.stocks[s]["width"] ) / (self.finish[f]["width"] * self.stocks[s]['weight']))
                # min of two max will satisfies both
                num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

                # make pattern and add to list of patterns
                if num_cuts > 0:
                    feasible = True
                    cuts_dict = {key: 0 for key in self.finish.keys()}
                    cuts_dict[f] = num_cuts
                    trim_loss = self.stocks[s]['width'] - sum([self.finish[f]["width"] * cuts_dict[f] for f in self.finish.keys()])
                    trim_loss_pct = round(trim_loss/self.stocks[s]['width'] * 100, 3)
                    self.patterns.append({"stock": s, "cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct })

            if not feasible:
                pass
                # print(f"No feasible pattern was found for Stock {s} and FG {f}")

        
if __name__ == "__main__":
    # LOAD CONFIG
    margin_df = pd.read_csv('data/min_margin.csv')
    spec_type = pd.read_csv('data/spec_type.csv')
    coil_priority = pd.read_csv('data/coil_data.csv')
    
    # INPUT
    fin_file_path = '20240710_finish_df.xlsx' # GLOBAL VAR
    # LOAD PARAM
    PARAMS = {'spec_name': 'JSH270C-PO',
              'type'    : "Carbon",
             'thickness': 2.6,
             'maker': 'POSCO',
             'code': 'POSCO JSH270C-PO 2.6'}
    sorted_sub_job_operator = [{'customer_name': 'VPIC1', 'total_need_cut': 23955.0},
                                {'customer_name': 'CIC', 'total_need_cut': 9357.123051681707}]

    df = pd.read_excel(fin_file_path)
    finish_df = df[
                    (df["spec_name"] == PARAMS["spec_name"]) & 
                    (df["thickness"] == PARAMS["thickness"]) &
                    (df["maker"] == PARAMS["maker"]) &
                    (df['need_cut'] < -10 ) 
                    ]
    
    cust = sorted_sub_job_operator[0]['customer_name']
    cust_f_df = finish_df[finish_df['customer_name']==cust]

    mc_file_path = "20240710_mc_df.xlsx"  # GLOBAL VAR
    df1 = pd.read_excel(mc_file_path)
    mc_df = df1[
                (df1["spec_name"] == PARAMS["spec_name"]) & 
                (df1["thickness"] == PARAMS["thickness"]) &
                (df1["maker"] == PARAMS["maker"]) 
                ]

    job1 = SteelCutting()
    job1.setup(cust_f_df, mc_df,PARAMS)
    job1.update_bound(3)
    job1.update_min_margin(margin_df)
    print(job1.stocks)