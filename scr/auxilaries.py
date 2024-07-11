from typing import Dict, Any
import copy
import pandas as pd
import numpy as np



def create_finish_demand_by_line_fr_naive_pattern(patterns, finish: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    finish {finish: width, need_cut, upper_bound,fc1,fc2,fc3 } 
    Convert demand in KGs to demand in slice on pool of stock
    """

    dump_ls = {}
    for f, finish_info in finish.items():
        try:
            non_zero_min = min([patterns[i]['cuts'][f] for i, _ in enumerate(patterns) if patterns[i]['cuts'][f] != 0])
        except ValueError:
            non_zero_min = 0
        dump_ls[f] = {**finish_info
                            ,"upper_demand_slice": max([patterns[i]['cuts'][f] for i,_ in enumerate(patterns)])
                            ,"demand_slice": non_zero_min }
    
    # Filtering the dictionary to include only items with keys in keys_to_keep
    new_finish_list = {k: v for k, v in dump_ls.items() if v['upper_demand_slice'] > 0} # xem lai dieu kien nay, tuc la neu cat dai nay voi stock hien co thì overcut lon

    return new_finish_list

def filter_stocks_by_width_and_weight(remain_stock, min_width=None, min_weight=None):
    """
    stocks {stock:receiving_date,width, weight, qty} 
    Filter remain_stock, which has the width and min_weight for each overused stock needed to be replaced
    """
    filtered_stocks = {}
    for stock_id, details in remain_stock.items():
        width = details['width']
        weight = details['weight']
        
        if ((min_width is None or width == min_width) and # dam bao trim loss khong bi thay doi lon
            (min_weight is None or weight >= min_weight)):
            filtered_stocks[stock_id] = details

    # Sort stocks by width and weight
    sorted_stocks = dict(
        sorted(
            filtered_stocks.items(),
            key=lambda item: (item[1]['width'], item[1]['weight'])
        )
    )
    
    return sorted_stocks

# FILTER STOCK OR PATTERN
def filter_out_stock_by_cond(stock_list, key):
    """
    Find stocks {stock:receiving_date,width, weight, qty} 
    with condition, take the list of pattern diff from the key
    """
    filtered_list = {}
    for s, stock_info in stock_list.items():
        if s != key:
            filtered_list[s] = {**stock_info}
    
    return filtered_list

def filter_out_pattern_by_cond(pattern_list, cond, key):
    """
    Find pattern {stock, cuts {}, trim_loss, trim_loss_pct} 
    with condition, take the list of pattern diff from the key
    """
    filtered_list = []
    for item in pattern_list:
        if item[cond] != key:
            filtered_list.append(item)
    
    return filtered_list

def filter_out_pattern_by_conds(pattern_list, cond1, cond2, key1,key2):
    """
    Find pattern {stock, cuts {}, trim_loss, trim_loss_pct} 
    with condition, take the list of pattern diff from the key
    """
    filtered_list = []
    for item in pattern_list:
        if item[cond1] == key1 or item[cond2] != key2: # khac stock overused, va giu id co trim loss thap
            filtered_list.append(item)

    # # Sort pattern by trim loss - largest to smallest
    # sorted_pattern = sorted(filtered_list,key=lambda x: x['trim_loss_pct'], reverse=True)
    
    return filtered_list


# COUNT FINAL CUT
def count_weight(data):
    # Initialize an empty dictionary for the total sums
    total_sums = {}
    
    # Loop through each dictionary in the list
    for entry in data:
        count = entry['count']
        cuts = entry['cut_w']
        
        # Loop through each key in the cuts dictionary
        for key, value in cuts.items():
            # If the key is not already in the total_sums dictionary, initialize it to 0
            if key not in total_sums:
                total_sums[key] = 0
            # Add the product of count and value to the corresponding key in the total_sums dictionary
            total_sums[key] += round(count * value,2)
    return total_sums

# Remove stock that produce too many patterns over order
def count_pattern(stock_list):
    """
    Count each stock is used how many times
    """
    
    stock_counts = {}

    # Iterate through the list and count occurrences of each stock
    for item in stock_list:
        stock = item['stock']
        count = 1
        if stock in stock_counts:
            stock_counts[stock] += count
        else:
            stock_counts[stock] = count
            
    return stock_counts