import pandas as pd
import numpy as np
    
def change_sign_numbers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = np.where(df[numeric_cols] < 0, -df[numeric_cols], np.abs(df[numeric_cols]))
    return df

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

