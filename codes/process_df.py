import pandas as pd
import numpy as np
from amplpy import AMPL
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

def create_upper_bound_need_cut(df, BOUND_VALUE):
    """
    Always recalculate with new need cut
    Minus need_cut will always right for DEFAULT & USER SETTING

    IMPROVE: co the chi can 1 cot upper bound
    """
    # Limited case only be in the consideration if NEED_CUT < 0:
    df['upper_bound_limited'] = np.where(df['need_cut'] < 0, 0.3 * df['fc1'] - df['need_cut'], np.nan)
    def create_fc_list(x):
        return [f"fc{i}" for i in range(1, x+1)]
    try:
        fc_columns = create_fc_list(BOUND_VALUE)

        if BOUND_VALUE <= 3:
            # Default allowing stock after cut < original need cut + X <=3 months forecast
            df['upper_bound_default'] = df[fc_columns].sum(axis=1) - df['need_cut']
        else:
            # User setting allowing stock after cut < original need cut + X<=6 months forecast
            df['upper_bound_user_setting'] =  df[fc_columns].sum(axis=1) - df['need_cut']
    except:
        pass
    
    return df

def filter_stock_df_to_dict(df, stock_id, value_columns, PRIORITY):

    # Sort data according to the priority of FIFO
    sorted_df = df.sort_values(by=['receiving_date','weight'], ascending=[True, False])

    # CASE 1: TRY TO OPTIMIZE WITH SEMI AND REWIND FIRST
    if PRIORITY == "CASE_1":
        sorted_df = sorted_df[sorted_df['status'].isin(['R:REWIND'
                                                        ,'Z:SEMI MCOIL'
                                                        ,'S:SEMI FINISHED']
                                                        )] # need standardization
    
    # CASE 2: WITH NORMAL MC HAVING TOTAL WEIGHT >> TOTAL NEED CUT 
    elif PRIORITY == "CASE_2":
        sorted_df = sorted_df[sorted_df['status'].isin(['M:RAW MATERIAL'
                                                        # ,'Z:SEMI MCOIL'
                                                        # ,'S:SEMI FINISHED'
                                                        ])] # need standardization
    else:
        pass # case cat lai FG or MC defective AND xu ly truong hop empty value

    # Set the index of the DataFrame to 'stock_id'
    sorted_df.set_index(stock_id, inplace=True)
    # Convert DataFrame to dictionary
    result = sorted_df[value_columns].to_dict(orient='index')

    return result

def filter_finish_excel_to_dict(df, finish_id, value_columns, BOUND_KEY, BOUND_VALUE):
    """
    Filter test case with finished goods have need_cut < 0 is the need_cut to be considered
    Note: 
    File path here is the new list of need cut/ AND / OR / list of stock after cut

    """
    # Create the upper bound for need cut
    filtered_df = create_upper_bound_need_cut(df, BOUND_VALUE)
    # Filter the DataFrame based on the upper bound column
    needcut_df = filtered_df[filtered_df['need_cut'] <= filtered_df[f"upper_bound_{BOUND_KEY}"]]

    # Sort DataFrame by 'need_cut'(negative) column in ascending order, 'width' column in descending order
    sorted_df = needcut_df.sort_values(by=['width','need_cut'], ascending=[False,True])

    # Initialize result dictionary - take time if the list long
    result = {f"F{int(row[finish_id])}": {column: row[column] for 
                                          column in value_columns} for 
                                          _, row in sorted_df.iterrows()}

    return result

def pandas_to_ampl_obj(df):
    # 1. Send the data from "amt_df" to AMPL and initialize the indexing set "FOOD"
    ampl.set_data(food_df, "FOOD")
    # 2. Send the data from "nutr_df" to AMPL and initialize the indexing set "NUTR"
    ampl.set_data(nutr_df, "NUTR")
    # 3. Set the values for the parameter "amt" using "amt_df"
    ampl.get_parameter("amt").set_values(amt_df)

if __name__ == "__main__":
    # file_path = "../data/test_mc_df.xlsx"
    # stock_key = "Inventory_ID"
    # value_columns = ["width", "weight"]
    # stocks = stock_excel_to_dict(file_path, stock_key, value_columns)
    # print(stocks)

    finish_file_path = "data/test_finish_df.xlsx"
    finish_key = "order_id"
    finish_value_columns = ["width", "need_cut","fc1"] 
    finish = filter_finish_excel_to_dict(finish_file_path,finish_key,finish_value_columns)
    print(finish)
