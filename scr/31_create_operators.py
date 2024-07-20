### PROCESS BY PARAMS ###
from data_management.process_df import filter_by_params, filter_multi_stock_df_to_dict
import pandas as pd
import numpy as np

### INPUT VARIABLE
param = 'POSCO+JSH270C-PO+2.6'
file_path = "../data/20240710_finish_df.xlsx"

# LOAD JOB
param_split = param.split("+")
maker = param_split[0]
spec = param_split[1]
thickness = round(float(param_split[2]),2)

PARAMS = {
    "spec_name" :spec,
    "thickness" :thickness,
    "maker"     : maker,
    "code"      : maker + " " + spec  + " " + str(thickness)
}

# Filter by PARAMS and need cut
param_df = filter_by_params(file_path, PARAMS)
param_df = param_df[param_df['need_cut'] < -10]
# CUSTOMER in the list
customer_list = param_df['customer_name'].unique()
sub_job_operator=[]
# finished goods list by customer
for cust in customer_list:
    cust_df = param_df[param_df['customer_name']==cust]
    total_needcut = sum(cust_df['need_cut']) * -1
    sub_job_operator.append({"customer_name": cust, 
                            "total_need_cut": float(total_needcut)})
    
    print(f'{cust}, Need cut {total_needcut}')


# KHACH co need cut nhieu hon uu tien cat trc
sorted_sub_job_operator = sorted(sub_job_operator, key=lambda x: x['total_need_cut'],reverse=True)

### OUTPUT
# sorted_sub_job_operator
# PARAMS
# param_df