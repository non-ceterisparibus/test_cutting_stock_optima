import pandas as pd
import numpy as np
from amplpy import AMPL
from typing import Dict, Any
# improve by polars

# Set Obj
ampl = AMPL()

def change_sign_numbers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = np.where(df[numeric_cols] < 0, -df[numeric_cols], np.abs(df[numeric_cols]))
    return df

def filter_by_params(file_path,params):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[(df["warehouse"] == params["warehouse"]) & 
                    (df["spec_name"] == params["spec_name"]) & 
                    (df["thickness"] == params["thickness"]) &
                    (df["maker"] == params["maker"])
                    ]
    return filtered_df

def query_by_params(file_path, params):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    # Store parameter values in variables
    warehouse_val = params["warehouse"]
    spec_name_val = params["spec_name"]
    thickness_val = params["thickness"]
    maker_val = params["maker"]
    
    # Use indexing if applicable
    # df.set_index('column_name', inplace=True)
    
    # Perform filtering using vectorized operations
    filtered_df = df.query(f'warehouse == "{warehouse_val}" & \
                            spec_name == "{spec_name_val}" & \
                            thickness == "{thickness_val}" & \
                            maker == "{maker_val}"')
    # Reset index if needed
    # filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df

def check_need_cut_qty(finish_df):
    # Check if 'column_name' has any values smaller than 0
    has_need_cut = (finish_df['need_cut'] < -50).any()

    if has_need_cut:
        total_need_cut = finish_df['need_cut'][finish_df['need_cut'] < -50]

        # Sum the negative values
        sum_of_total_need_cuts = -total_need_cut.sum()
        return sum_of_total_need_cuts
    else:
        print("There is no need cut")

def choose_stocks_by_need_cuts(df, sum_of_total_need_cuts, over_cut_rate):
    # Initialize variables
    cumulative_sum = 0
    rows_to_include = 0

    # Iterate through the DataFrame rows
    for value in df['weight']:
        cumulative_sum += value
        rows_to_include += 1
        if cumulative_sum > sum_of_total_need_cuts *(1+over_cut_rate):
            break

    # Filter the DataFrame to include the first few rows
    try:
        filtered_df = df.iloc[:rows_to_include]
    except IndexError:
        filtered_df = df
    return filtered_df

def choose_stock_by_status(df):
    result = df['status'].isin(['R:REWIND'
                            ,'Z:SEMI MCOIL'
                            ,'S:SEMI FINISHED'])
    if any(result):
        sorted_df = df[df['status'].isin(['R:REWIND'
                                        ,'Z:SEMI MCOIL'
                                        ,'S:SEMI FINISHED']
                                        )] # need standardization
    else:
        sorted_df = df[df['status'].isin(['M:RAW MATERIAL'
                                                        # ,'Z:SEMI MCOIL'
                                                        # ,'S:SEMI FINISHED'
                                                        ])] # need standardization
    return sorted_df


def filter_stock_df_to_dict(df, stock_id, value_columns, sum_of_total_need_cuts):

    # Sort data according to the priority of FIFO
    sorted_df = df.sort_values(by=['receiving_date','weight'], ascending=[True, False])

    sorted_df = choose_stock_by_status(sorted_df)
    sorted_df = choose_stocks_by_need_cuts(sorted_df, sum_of_total_need_cuts, over_cut_rate = 0.3)
    # Set the index of the DataFrame to 'stock_id'
    sorted_df.set_index(stock_id, inplace=True)
    # Convert DataFrame to dictionary
    result = sorted_df[value_columns].to_dict(orient='index')

    return result

def create_upper_bound_need_cut(df, BOUND_VALUE=None):
    """
    Always recalculate with new need cut
    Minus need_cut will always right for DEFAULT & USER SETTING

    IMPROVE: co the chi can 1 cot upper bound
    """
    # Limited case only be in the consideration if NEED_CUT < 0:
    def create_fc_list(x):
        return [f"fc{i}" for i in range(1, x+1)]
    
    if BOUND_VALUE is not None:
        fc_columns = create_fc_list(BOUND_VALUE)

        if BOUND_VALUE <= 3:
            # Default allowing stock after cut < original need cut + X <=3 months forecast
            df['upper_bound_default'] = df[fc_columns].sum(axis=1) - df['need_cut']
        else:
            # User setting allowing stock after cut < original need cut + X<=6 months forecast
            df['upper_bound_user_setting'] =  df[fc_columns].sum(axis=1) - df['need_cut']

        df['upper_bound_po'] = df[fc_columns].sum(axis=1)
    else:
        df['upper_bound_limited'] = np.where(df['need_cut'] < 0, 0.3 * df['fc1'] - df['need_cut'], np.nan)
        df['upper_bound_po']  = df['upper_bound_limited'] 
    
    return df

def check_finish_weight_per_stock(weight_s: float, width_s: float, finish: Dict[str, Dict[str, Any]], BOUND_KEY: str) -> Dict[str, bool]:
    wu = weight_s / width_s # MC WEIGHT PER UNIT
    weight_f = {f: finish[f]["width"]*wu for f in finish.keys()}
    f_upper_demand = {f: finish[f][f"upper_bound_{BOUND_KEY}"] for f in finish.keys()}

    check_f ={f: weight_f[f] < f_upper_demand[f] for f in finish.keys()}

    return check_f

def filter_finish_df_to_dict(df, finish_id, value_columns, BOUND_VALUE):
    """
    Filter test case with finished goods have need_cut < 0 is the need_cut to be considered
    Note: 
    File path here is the new list of need cut/ AND / OR / list of stock after cut

    """
    # Create the upper bound for need cut
    filtered_df = create_upper_bound_need_cut(df, BOUND_VALUE)
    # Filter the DataFrame based on the upper bound column (always right even need cut > 0)
    needcut_df = filtered_df[filtered_df['need_cut'] <= filtered_df["upper_bound_po"]]

    # Sort DataFrame by 'need_cut'(negative) column in ascending order, 'width' column in descending order
    sorted_df = needcut_df.sort_values(by=['width','need_cut'], ascending=[False,True])

    # Initialize result dictionary - take time if the list long
    result = {f"F{int(row[finish_id])}": {column: row[column] for 
                                          column in value_columns} for 
                                          _, row in sorted_df.iterrows()}

    return result

def filter_stock_df(df, stock_id, value_columns):
    status_conditions = df['status'].isin(['R:REWIND'
                            ,'Z:SEMI MCOIL'
                            ,'S:SEMI FINISHED'])
    
    # Sort data according to the priority of FIFO
    sorted_df = df.sort_values(by=['receiving_date', 'weight'], ascending=[True, False])

    # Filter dataframe based on status conditions
    if any(status_conditions):
        sorted_df = sorted_df[sorted_df['status'].isin(['R:REWIND', 'Z:SEMI MCOIL', 'S:SEMI FINISHED'])]
    else:
        sorted_df = sorted_df[sorted_df['status'].isin(['M:RAW MATERIAL'])]

    # Set the index of the DataFrame to 'stock_id'
    sorted_df.set_index(stock_id, inplace=True)
    
    # Select desired columns
    result = sorted_df[value_columns]

    return result

if __name__ == "__main__":
    # file_path = "../data/test_mc_df.xlsx"
    # stock_key = "Inventory_ID"
    # value_columns = ["width", "weight"]
    # stocks = stock_excel_to_dict(file_path, stock_key, value_columns)
    # print(stocks)

    finish_file_path = "data/test_finish_df.xlsx"
    finish_key = "order_id"
    finish_value_columns = ["width", "need_cut","fc1"] 
    finish = filter_finish_df_to_dict(finish_file_path,finish_key,finish_value_columns)
    print(finish)
