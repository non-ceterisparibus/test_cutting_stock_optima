from gurobipy import *
import pandas as pd
import numpy as np

def multi_dict():
  number_of_vehicles = 2 
  vehicles_origin = [4, 6]
  vehicles_destination = [3, 0]
  total_time_vehicle = [35, 27]

  truck = [i for i in range(number_of_vehicles)]
  starting_node = {}
  destination_nodes = {}
  time = {}
  for i in truck:
    starting_node[i] = vehicles_origin[i]
    destination_nodes[i] = vehicles_destination[i]
    time[i] = total_time_vehicle[i]


  multi = {}
  for i in range(number_of_vehicles):
    l = [vehicles_origin[i],vehicles_destination[i],total_time_vehicle[i]]
    multi[i] = l

  truck, starting_node, destination_nodes, time = multidict(multi)
  return 
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