import pandas as pd
import numpy as np
import json
import datetime
import copy
import os

# INPUT
fin_file_path = os.getenv('FIN_DF_PATH')
mc_file_path = os.getenv('MC_DF_PATH')
stock_ratio_default = float(os.getenv('STOCK_RATIO_DEFAULT', '0.5'))

# DATA
spec_type_df = pd.read_csv('scr/model_config/spec_type.csv')

# SETUP
finish_key = 'order_id'
finish_columns = [
                "customer_name",
                # "customer",
                "width", "need_cut", "Standard",
                "fc1", "fc2", "fc3", "average FC",
                "1st Priority", "2nd Priority", "3rd Priority",
                "Min_weight", "Max_weight"
                  ]
forecast_columns = ["fc1", "fc2", "fc3"]

stock_key = "inventory_id"
stock_columns = ['receiving_date',"width", "weight",'warehouse',
                 "status",'remark']

def div(numerator, denominator):
    def division_operation(row):
        if row[denominator] == 0:
            if row[numerator] > 0:
                return np.inf
            elif row[numerator] < 0:
                return -np.inf
            else:
                return np.nan  # Handle division by zero with numerator equal to zero
        else:
            return float(row[numerator] / row[denominator])
    return division_operation

def create_finish_dict(finish_df):
  
    finish_df.loc[:, 'stock_ratio'] = finish_df.apply(div('need_cut', 'average FC'), axis=1)
    can_cut_df = finish_df[finish_df['stock_ratio'] < float(stock_ratio_default)] # chon ca nhung need cut ko am
    
    # Width - Decreasing// need_cut - Descreasing // Average FC - Increasing
    sorted_df = can_cut_df.sort_values(by=['need_cut','width'], ascending=[True,False]) # need cut van dang am
    # sorted_df = can_cut_df.sort_values(by=['width','need_cut'], ascending=[False,True,True]) # need cut van dang am
    
    # Fill NaN values in specific columns with the average, ignoring NaN
    sorted_df[finish_columns] = sorted_df[finish_columns].fillna("")
    # sorted_df[forecast_columns] = sorted_df[forecast_columns].apply(lambda x: x.fillna(x.mean()), axis=1)
    sorted_df[forecast_columns] = sorted_df[forecast_columns].fillna(0)

    # Initialize result dictionary - take time if the list long
    finish = {f"F{int(row[finish_key])}": {column: row[column] for 
                                          column in finish_columns} for 
                                          _, row in sorted_df.iterrows()}
    return finish

def create_stocks_dict(stock_df):
    # Sort data according to the priority of FIFO
    sorted_df = stock_df.sort_values(by=['warehouse','weight','receiving_date'], ascending=[True,True, True])
    
    # Set the index of the DataFrame to 'inventory_id'
    sorted_df.set_index(stock_key, inplace=True)
    
    # Convert DataFrame to dictionary
    sorted_df[stock_columns] = sorted_df[stock_columns].fillna("")
    stocks = sorted_df[stock_columns].to_dict(orient='index')
   
    return stocks

def filter_by_materialprops(file_path, materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    (df["spec_name"] == materialprops["spec_name"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    return filtered_df

def filter_finish_by_materialprops(file_path, materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    (df["customer_name"] == materialprops["customer"]) & 
                    #  (df["customer"] == materialprops["customer"]) &
                    (df["spec_name"] == materialprops["spec_name"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    return filtered_df

def find_spec_type(spec,spec_type_df):
    try:
        type = spec_type_df[spec_type_df['spec']==spec]['type'].values[0]
    except IndexError:
        type = "All"
    return type

# PROCESS
if __name__ == "__main__":
    
    today = datetime.datetime.today()
    formatted_date = today.strftime("%y-%m-%d")

    finish_list = {
        'date': formatted_date,
        'materialprop_finish':{}
    }
    stocks_list = {
        'date': formatted_date,
        'materialprop_stock':{}
    }

    with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
        job_list = json.load(file)
    
    # print(f"use stock ratio {stock_ratio_default}")

    for job in job_list['jobs']: 
        # loop to create stock list
        MATERIALPROPS = {}
        materialprop = job['materialprop']
        materialprop_split = materialprop.split("+")
        maker = materialprop_split[0]
        spec = materialprop_split[1]
        thickness = round(float(materialprop_split[2]),2)

        MATERIALPROPS = {
                    "spec_name" :spec,
                    "type"      : find_spec_type(spec,spec_type_df),
                    "thickness" :thickness,
                    "maker"     : maker,
                    "code"      : maker + " " + spec  + " " + str(thickness)
                    }

        mc_df = filter_by_materialprops(mc_file_path, MATERIALPROPS)
        stocks = create_stocks_dict(mc_df)

        stocks_list['materialprop_stock'][materialprop] = {'materialprop': MATERIALPROPS, 'stocks': stocks}
        finish_list['materialprop_finish'][materialprop] = {'materialprop': MATERIALPROPS, 'customer':[]}
        for cust, value in job['tasks'].items():
            # loop to create finish list accordingly with materialprop and customer
            MATERIALPROPS1 = {
                    'customer'  :cust,
                    "spec_name" :spec,
                    "type"      : find_spec_type(spec,spec_type_df),
                    "thickness" :thickness,
                    "maker"     : maker,
                    "code"      : maker + " " + spec  + " " + str(thickness)
                    }
            finish_df = filter_finish_by_materialprops(fin_file_path, MATERIALPROPS1)
            finish = create_finish_dict(finish_df)
            finish_list['materialprop_finish'][materialprop]['customer'].append({cust: finish})
        
    with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'w') as stocks_file:
        json.dump(stocks_list, stocks_file, indent=3)

    with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'w') as json_file:
        json.dump(finish_list, json_file, indent=3)
