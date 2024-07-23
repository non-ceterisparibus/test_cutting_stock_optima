# GET OVERLAP PARAMS
import pandas as pd
import numpy as np
import json
import datetime
import copy

# INPUT
fin_file_path = "data/20240710_finish_df.xlsx"
mc_file_path = "data/20240710_mc_df.xlsx"

finish_key = 'order_id'
finish_columns = ["customer_name","width", "need_cut", "fc1", "fc2", "fc3"]
stock_key = "inventory_id"
stock_columns = ['receiving_date',"width", "weight",'warehouse']

def filter_by_params(file_path,params):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    (df["spec_name"] == params["spec_name"]) & 
                    (df["thickness"] == params["thickness"]) &
                    (df["maker"] == params["maker"])
                    ]
    return filtered_df


# PROCESS
fin_df = pd.read_excel(fin_file_path)
mc_df = pd.read_excel(mc_file_path)

has_need_cut_df = fin_df[fin_df['need_cut'] < -10]

has_need_cut_df.loc[:, 'params'] = (
    has_need_cut_df['maker'] + "+" +
    has_need_cut_df['spec_name'] + "+" +
    has_need_cut_df['thickness'].astype(str)
    )

fin_params = has_need_cut_df['params'].unique()

mc_df.loc[:, 'params'] = (mc_df['maker'] + "+" + mc_df['spec_name']+ "+" + mc_df['thickness'].astype(str))
mc_params = mc_df['params'].unique()

# Find the intersection (overlapping values)
overlap = set(mc_params) & set(fin_params)

# Convert the result back to a list if needed
params = list(overlap)
n_jobs = len(params)

today = datetime.datetime.today()
# Format the date to dd/mm/yy
formatted_date = today.strftime("%d-%m-%y")

job_list = {
    'date': formatted_date,
    'number of job': n_jobs,
    'jobs':[

    ]
}
for i, param in enumerate(params):
    # LOAD JOB
    # print(f"process param {param}")
    
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

    # Filter FINISH by PARAMS and need cut
    param_finish_df = filter_by_params(fin_file_path, PARAMS)
    to_cut_finish_df = param_finish_df[param_finish_df['need_cut'] < -10]

    # Filter STOCKS by PARAMS 
    param_stocks_df = filter_by_params(mc_file_path, PARAMS)
    sum_stock = param_stocks_df['weight'].sum()

    job_list['jobs'].append({'param': param,'stocks_available': float(sum_stock) ,'tasks':{}})
    current_job = job_list['jobs'][i]
    # CUSTOMER in the list
    customer_list = to_cut_finish_df['customer_name'].unique()
    sub_job_operator = {}
    # finished goods list by customer
    for cust in customer_list:
        cust_df = to_cut_finish_df[to_cut_finish_df['customer_name']==cust]
        total_needcut = sum(cust_df['need_cut']) * -1
        sub_job_operator[cust] = {"total_need_cut": float(total_needcut)}

    # KHACH co need cut nhieu hon uu tien cat trc
    sorted_data = dict(sorted(sub_job_operator.items(), key=lambda item: item[1]['total_need_cut'], reverse=True))
    current_job['tasks'] = copy.deepcopy(sorted_data)


with open(f'job-list-{formatted_date}.json', 'w') as json_file:
    json.dump(job_list, json_file, indent=4)