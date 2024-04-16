import pandas as pd
import numpy as np
    

def stock_excel_to_dict(file_path, key_column, value_columns):
    df = pd.read_excel(file_path)
    result = {}

    for _, row in df.iterrows():
        key = row[key_column]
        values = {column: row[column] for column in value_columns}
        result[key] = values

    return result

def finish_excel_to_dict(file_path, key_column, value_columns):
    df = pd.read_excel(file_path)
    result = {}

    for _, row in df.iterrows():
        fkey = row[key_column]
        values = {column: row[column] for column in value_columns}
        result[f"F{fkey}"] = values

    return result

if __name__ == "__main__":
    # file_path = "data/test_mc_df.xlsx"
    # stock_column = "Inventory_ID"
    # value_columns = ["width", "weight"]
    # stocks = stock_excel_to_dict(file_path, stock_column, value_columns)
    # print(stocks)

    finish_file_path = "data/test_finish_df.xlsx"
    finish_column = "Order_ID"
    finish_value_columns = ["width", "need_cut","fc1"] 
    finish = finish_excel_to_dict(finish_file_path,finish_column,finish_value_columns)
    print(finish)

