# GET OVERLAP MATERIAL_PROPERTIES
import pandas as pd
import numpy as np
import json
import datetime
import math
import copy
import os
import re

global stock_ratio_default

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
                "width", "need_cut", "standard",
                "fc1", "fc2", "fc3", "average FC",
                "1st Priority", "2nd Priority", "3rd Priority",
                "Min_weight", "Max_weight", "Min_MC_weight"
                  ]
forecast_columns = ["fc1", "fc2", "fc3", "average FC"]

stock_key = "inventory_id"
stock_columns = ['receiving_date',"width", "weight",'warehouse',
                 "status",'remark']

def find_spec_type(spec,spec_type_df):
    try:
        type = spec_type_df[spec_type_df['spec']==spec]['type'].values[0]
    except IndexError:
        type = "All"
    return type

def find_materialprops_and_jobs(fin_file_path,mc_file_path):
    """_Find overlapped material code with need cut < 0_

    Args:
        fin_file_path (_type_): _description_
        mc_file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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

def group_cut_standard(df):
    # DF only have need-cut < 0
    custom_order =["small","small-medium","medium","medium-big","big"]
    df["standard"] = pd.Categorical(df["standard"], categories=custom_order, ordered=True)
    
    df.loc[:,'cut_standard'] = df["standard"].apply(
    lambda x: "small-medium" if x in ["small-medium","medium", "small"] else "medium-big"
    )
    
    # Create new empty df to save split group
    appended_df = pd.DataFrame(columns=df.columns)
    std_dict = {}
    for std in ["small-medium","medium-big"]:
        filtered_df = df[df['cut_standard'] == std]
        
        sorted_desc = filtered_df.sort_values(by="width", ascending=False)  # Largest first
        sorted_asc = filtered_df.sort_values(by="width", ascending=True)   # Smallest first
        
        count = filtered_df.shape[0]
        # Interleave rows
        interleaved_rows = []
        
        # chia thanh nhieu nhom neu co qua nhieu fg code
        if count <= 15:
            appended_df = pd.concat([appended_df, filtered_df], ignore_index=True)
            
        else:
            n = math.ceil(count/15) #n group
            num_it = int(count//n) # each group have num item
            
            # Create interleaved df
            gr2 = int(count//2)
            for i in range(gr2):
                interleaved_rows.append(sorted_desc.iloc[i])  # Add largest
                interleaved_rows.append(sorted_asc.iloc[i])  # Add smallest

            # Handle odd-length DataFrame (optional, here for completeness)
            if count % 2 != 0:
                interleaved_rows.append(sorted_desc.iloc[gr2])

            # Convert interleaved rows to a DataFrame
            interleaved_df = pd.DataFrame(columns=df.columns,data=interleaved_rows)

            print(f"split tb {count} {n}")
            # Split into two DataFrames
            interleaved_df = interleaved_df.reset_index(drop=True)
            for i in range(n):
                if i != n-1:
                    interleaved_df.loc[num_it*(i):num_it*(i+1),'cut_standard'] = f"{std}{str((i+1))}"    # Rows from `num` onwards
                else:
                    interleaved_df.loc[num_it*(i):,'cut_standard'] = f"{std}{str((i+1))}"
                    std_dict[std] = i
                    
            appended_df = pd.concat([appended_df, interleaved_df], ignore_index=True)
    
    return appended_df, std_dict

def group_cut_defined_standard(df, defined_std, std_dict): 
    # DF by material code
    custom_order =["small","small-medium","medium","medium-big","big"]
    df["standard"] = pd.Categorical(df["standard"], categories=custom_order, ordered=True)
    
    df.loc[:,'cut_standard'] = df["standard"].apply(
    lambda x: "small-medium" if x in ["small-medium","medium", "small"] else "medium-big"
    )
    
    # Step 1:
    # defined_std = ['small-medium1', 'small-medium2','medium-big']
    sub_group = [item for item in defined_std if re.search(r'\d', item)]
    re_group = [item for item in defined_std if not re.search(r'\d', item)] #remained
    
    # Step 2: Remove numbers from the filtered items
    cleaned_list = [re.sub(r'\d', '', item) for item in sub_group]

    # Step 3: Create a unique list
    unique_sub_list = list(set(cleaned_list))
    
    if not unique_sub_list: # new ko co sub group
        appended_df = copy.deepcopy(df)
    else:
        if len(re_group) == 0:
            # Create new empty df to save split group
            appended_df = pd.DataFrame(columns=df.columns)
        else:
            appended_df = copy.deepcopy(df[df['cut_standard']==str(re_group[0])])
            
        for std in unique_sub_list:
            # default: chia doi list
            filtered_df = df[df['cut_standard'] == std]
            sorted_desc = filtered_df.sort_values(by="width", ascending=False)  # Largest first
            sorted_asc = filtered_df.sort_values(by="width", ascending=True)   # Smallest first
            count = filtered_df.shape[0]
            
            # Interleave rows
            interleaved_rows = []
            
            # chia thanh nhieu nhom neu co qua nhieu fg code
            num = int(count//2)
            
            for i in range(num):
                interleaved_rows.append(sorted_desc.iloc[i])  # Add largest
                interleaved_rows.append(sorted_asc.iloc[i])  # Add smallest
            # Handle odd-length DataFrame (optional, here for completeness)
            if count % 2 != 0:
                interleaved_rows.append(sorted_desc.iloc[num])
                
            # Convert interleaved rows to a DataFrame
            interleaved_df = pd.DataFrame(columns=df.columns,data=interleaved_rows)
            
            print(f"def standard {std} {defined_std}")
            n= std_dict[std] # number of group
            num_it = int(count//n) # item per group
            
            # Split into two DataFrames
            interleaved_df = interleaved_df.reset_index(drop=True)
            # interleaved_df.loc[:num,'cut_standard'] = f"{std}1"  # First `num` rows
            # interleaved_df.loc[num:,'cut_standard'] = f"{std}2"    # Rows from `num` onwards
            for i in range(n):
                if i != n-1:
                    interleaved_df.loc[num_it*(i):num_it*(i+1),'cut_standard'] = f"{std}{str((i+1))}"    # Rows from `num` onwards
                else:
                    interleaved_df.loc[num_it*(i):,'cut_standard'] = f"{std}{str((i+1))}"

            appended_df = pd.concat([appended_df, interleaved_df], ignore_index=True)
        
    return appended_df

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
    # Width - Decreasing// need_cut - Descreasing // Average FC - Increasing
    sorted_df = finish_df.sort_values(by=['need_cut','width'], ascending=[True,False]) # need cut van dang am

    sorted_df[["Min_weight", "Max_weight"]] = sorted_df[["Min_weight", "Max_weight"]].fillna("")
    
    # Fill NaN values in specific columns with the average, ignoring NaN
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

def filter_finish_by_mat_stock_ratio(file_path, materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    # (df["customer_name"] == materialprops["customer"]) & 
                    (df["spec_name"] == materialprops["spec_name"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    filtered_df.loc[:, 'stock_ratio'] = filtered_df.apply(div('need_cut', 'average FC'), axis=1)
    df = filtered_df[filtered_df['stock_ratio'] < float(stock_ratio_default)] # chon ca nhung need cut ko am
    
    return df

if __name__ == "__main__":
    materialprops, n_jobs = find_materialprops_and_jobs(fin_file_path =fin_file_path,
                                                        mc_file_path = mc_file_path)

    # DATE
    today = datetime.datetime.today()
    formatted_date = today.strftime("%y-%m-%d")

    job_list = {
        'date': formatted_date,
        'number of job': n_jobs,
        'jobs':[]
    }

    finish_list = {
        'date': formatted_date,
        'materialprop_finish':{}
    }
    stocks_list = {
        'date': formatted_date,
        'materialprop_stock':{}
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
            "type"      :find_spec_type(spec,spec_type_df),
            "code"      : HTV_code
        }
        
        finish_list['materialprop_finish'][materialprop] = {'materialprop': MATERIALPROPS, 'group':[]}

        # Filter FINISH by MATERIAL_PROPERTIES 
        materialprop_finish_df = filter_finish_by_mat_stock_ratio(fin_file_path, MATERIALPROPS)
        
        # Take unique list with need-cut < 0
        to_cut_finish_df = materialprop_finish_df[materialprop_finish_df['need_cut'] < 0]
        defined_standard = to_cut_finish_df['standard'].unique().tolist()
        desired_order = ["small","small-medium", "big", "medium"] 
        # Reorder dynamically based on the desired order
        ordered_standard = sorted(defined_standard, key=lambda x: desired_order.index(x))
        
        # Create FINISH list with customer group
        for cust_gr in ordered_standard:
            filtered_finish_df = materialprop_finish_df[materialprop_finish_df['standard']==cust_gr]
            finish = create_finish_dict(filtered_finish_df)
            finish_list['materialprop_finish'][materialprop]['group'].append({cust_gr: finish})
        
        # Filter STOCKS by MATERIAL_PROPERTIES
        materialprop_stocks_df = filter_by_materialprops(mc_file_path, MATERIALPROPS)
        stocks = create_stocks_dict(materialprop_stocks_df)
        # Add to stocks lists
        stocks_list['materialprop_stock'][materialprop] = {'materialprop': MATERIALPROPS, 'stocks': stocks}
        
        # Add to JOB lists
        job_list['jobs'].append({"job": i,'materialprop': materialprop,"code": HTV_code, 'stocks_available': {} ,'tasks':{}})
        current_job = job_list['jobs'][i]
        
        # SUM STOCK BY WAREHOUSE
        wh_list = materialprop_stocks_df['warehouse'].unique().tolist()
        for wh in wh_list:
            wh_stock = materialprop_stocks_df[materialprop_stocks_df['warehouse'] == wh]
            sum_wh_stock = wh_stock['weight'].sum()
            current_job['stocks_available'][wh] = float(sum_wh_stock)
            
        sub_job_operator = {}
        total_need_cut = 0

        # finished goods list by cut group
        for std_group in defined_standard:
            cust_df = to_cut_finish_df[to_cut_finish_df['standard']==std_group]
            
            needcut = sum(cust_df['need_cut']) * -1
            sub_job_operator[std_group] = {"total_need_cut": float(needcut)}
            total_need_cut += needcut

        # KHACH co need cut nhieu hon uu tien cat trc
        sorted_data = dict(sorted(sub_job_operator.items(), key=lambda item: item[1]['total_need_cut'], reverse=True))
        current_job['tasks'] = copy.deepcopy(sorted_data)
        current_job['total_need_cut'] = total_need_cut

    with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'w') as json_file:
        json.dump(job_list, json_file, indent=4)
       
    with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'w') as stocks_file:
        json.dump(stocks_list, stocks_file, indent=3)

    with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'w') as json_file:
        json.dump(finish_list, json_file, indent=3)
