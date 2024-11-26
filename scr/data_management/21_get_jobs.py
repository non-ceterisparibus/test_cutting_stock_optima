# GET OVERLAP MATERIAL_PROPERTIES
import pandas as pd
import numpy as np
import json
import datetime
import copy
import os

# INPUT
fin_file_path = os.getenv('FIN_DF_PATH')
mc_file_path = os.getenv('MC_DF_PATH')

def filter_by_materialprops(file_path,materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    (df["spec_name"] == materialprops["spec_name"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    return filtered_df

def load_dta(fin_file_path,mc_file_path):
    # LOAD DATA
    fin_df = pd.read_excel(fin_file_path)
    mc_df = pd.read_excel(mc_file_path)

    has_need_cut_df = fin_df[fin_df['need_cut'] < -10]

    has_need_cut_df.loc[:, 'materialprops'] = (
        has_need_cut_df['maker'] + "+" +
        has_need_cut_df['spec_name'] + "+" +
        has_need_cut_df['thickness'].astype(str)
        )

    fin_materialprops = has_need_cut_df['materialprops'].unique()

    mc_df.loc[:, 'materialprops'] = (mc_df['maker'] + "+" + mc_df['spec_name']+ "+" + mc_df['thickness'].astype(str))
    mc_materialprops = mc_df['materialprops'].unique()

    # Find the intersection (overlapping values)
    overlap = set(mc_materialprops) & set(fin_materialprops)

    # Convert the result back to a list if needed
    materialprops = sorted(list(overlap))
    n_jobs = len(materialprops)
    
    return materialprops, n_jobs

if __name__ == "__main__":
    materialprops, n_jobs = load_dta(fin_file_path =fin_file_path,
                              mc_file_path = mc_file_path)

    # DATE
    today = datetime.datetime.today()
    formatted_date = today.strftime("%y-%m-%d")

    job_list = {
        'date': formatted_date,
        'number of job': n_jobs,
        'jobs':[

        ]
    }
    for i, materialprop in enumerate(materialprops):
        # LOAD JOB
        materialprop_split = materialprop.split("+")
        maker = materialprop_split[0]
        spec = materialprop_split[1]
        thickness = round(float(materialprop_split[2]),2)
        HTV_code = maker + " " + spec  + " " + str(thickness)

        MATERIALPROPS = {
            "spec_name" :spec,
            "thickness" :thickness,
            "maker"     : maker,
            "code"      : HTV_code
        }

        # Filter FINISH by MATERIAL_PROPERTIES and need cut
        materialprop_finish_df = filter_by_materialprops(fin_file_path, MATERIALPROPS)
        to_cut_finish_df = materialprop_finish_df[materialprop_finish_df['need_cut'] < 0]

        # Filter STOCKS by MATERIAL_PROPERTIES
        materialprop_stocks_df = filter_by_materialprops(mc_file_path, MATERIALPROPS)
        job_list['jobs'].append({"job": i,'materialprop': materialprop,"code": HTV_code, 'stocks_available': {} ,'tasks':{}})
        current_job = job_list['jobs'][i]
        
        # SUM STOCK BY WAREHOUSE
        wh_list = materialprop_stocks_df['warehouse'].unique().tolist()
        for wh in wh_list:
            wh_stock = materialprop_stocks_df[materialprop_stocks_df['warehouse'] == wh]
            sum_wh_stock = wh_stock['weight'].sum()
            current_job['stocks_available'][wh] = float(sum_wh_stock)
            
        # CUSTOMER in the list
        customer_list = to_cut_finish_df['customer_name'].unique()
        # customer_list = to_cut_finish_df['customer'].unique()
        sub_job_operator = {}
        total_need_cut = 0
        # finished goods list by customer
        for cust in customer_list:
            cust_df = to_cut_finish_df[to_cut_finish_df['customer_name']==cust]
            # cust_df = to_cut_finish_df[to_cut_finish_df['customer']==cust]
            needcut = sum(cust_df['need_cut']) * -1
            sub_job_operator[cust] = {"total_need_cut": float(needcut)}
            total_need_cut += needcut

        # KHACH co need cut nhieu hon uu tien cat trc
        sorted_data = dict(sorted(sub_job_operator.items(), key=lambda item: item[1]['total_need_cut'], reverse=True))
        current_job['tasks'] = copy.deepcopy(sorted_data)
        current_job['total_need_cut'] = total_need_cut


    with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'w') as json_file:
        json.dump(job_list, json_file, indent=4)