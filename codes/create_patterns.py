import pandas as pd
import numpy as np
import os

def make_patterns_by_weight_width(stocks, finish, BOUND_KEY, MIN_MARGIN):
    """
    Generates patterns of feasible cuts from stock widths to meet specified finish widths.

    Parameters:
    stocks (dict): A dictionary where keys are stock identifiers and values are dictionaries
                   with key 'width' representing the width of each stock.

    finish (dict): A dictionary where keys are finish identifiers and values are dictionaries
                   with key 'width' representing the required finish widths.
    Returns:
    patterns (list): A list of dictionaries, where each dictionary represents a pattern of cuts.
                   Each pattern dictionary contains 'stock' (the stock identifier) and 'cuts'
                   (a dictionary where keys are finish identifiers and the value is the number
                   of cuts from the stock for each finish).
                   

                   Naive pattern with maximum number of cuts of each Finished Goods
                   that is closet to the required need_cut
                   and SUM(widths of FG) smaller Mother Coil width
    """
    patterns = []
    for f in finish:
        feasible = False

        for s in stocks:
            # max number of f that fit on s 
            num_cuts_by_width = int((stocks[s]["width"]-MIN_MARGIN) / finish[f]["width"])
            # max number of f that satisfied the need cut WEIGHT BOUND
            num_cuts_by_weight = int((finish[f][f"upper_bound_{BOUND_KEY}"] * stocks[s]["width"] ) / (finish[f]["width"]*stocks[s]['weight']))
            # min of two max will satisfies both
            num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

            # make pattern and add to list of patterns
            if num_cuts > 0:
                feasible = True
                cuts_dict = {key: 0 for key in finish.keys()}
                cuts_dict[f] = num_cuts
                patterns.append({"stock": s
                                ,"cuts": cuts_dict})
        if not feasible:
            print(f"No feasible pattern was found for Stock {s} and FG {f}")
            # return []

    return patterns

def make_patterns_by_width(stocks, finish, MIN_MARGIN):
    """
    Generates patterns of feasible cuts from stock widths to meet specified finish widths.

    Parameters:
    stocks (dict): A dictionary where keys are stock identifiers and values are dictionaries
                   with key 'width' representing the width of each stock.

    finish (dict): A dictionary where keys are finish identifiers and values are dictionaries
                   with key 'width' representing the required finish widths.

    Returns:
    patterns (list): A list of dictionaries, where each dictionary represents a pattern of cuts.
                   Each pattern dictionary contains 'stock' (the stock identifier) and 'cuts'
                   (a dictionary where keys are finish identifiers and the value is the number
                   of cuts from the stock for each finish).

                   Naive pattern with maximum number of cuts that is closet to the Mother Coil width
    """

    if MIN_MARGIN is None:
        MIN_MARGIN == 8
    else:
        pass

    patterns = []
    for f in finish:
        feasible = False

        for s in stocks:
            # max number of f that fit on s 
            num_cuts = int((stocks[s]["width"]-MIN_MARGIN) / finish[f]["width"])

            # make pattern and add to list of patterns
            if num_cuts > 0:
                feasible = True
                cuts_dict = {key: 0 for key in finish.keys()}
                cuts_dict[f] = num_cuts
                patterns.append({"stock": s
                                 ,"cuts": cuts_dict
                                 })
        if not feasible:
            print(f"No feasible pattern was found for Stock {s} and FG {f}")
            # return []

    return patterns

def ap_upper_bound(naive_patterns,finish):
    #upper bound of possible cut over all kind of stocks
    return {f: max([naive_patterns[i]['cuts'][f] for i,_ in enumerate(naive_patterns)]) for f in finish.keys()}

def change_order_dict(input_dict):
    """
    To reorder the keys such that those with a value of 0 are moved to the bottom 
    while preserving the original order of the other keys
    """
    # Extract keys with value 0
    keys_with_value_0 = [key for key, value in input_dict.items() if value == 0]

    # Remove keys with value 0 from the dictionary
    remaining_keys = {key: input_dict[key] for key in input_dict if key not in keys_with_value_0}

    # Add keys with value 0 to the end
    ordered_dict = {**remaining_keys, **{key: input_dict[key] for key in keys_with_value_0}}
    return ordered_dict


def ap_stock_bound(naive_patterns, finish, s):
    """
    Upper bound of possible cuts for each stock s

    Parameters:
        naive_patterns: A list of dictionaries containing pattern information.
        finish: A dictionary containing information about different finishes.
        s: An integer representing the stock value.

        Return Value:
        A tuple containing two dictionaries:
            max_cuts_dict: A dictionary with finish types as keys and maximum number of cuts as values.
            min_cuts_dict: A dictionary with finish types as keys and minimum number of cuts as values.
    """
    filtered_patterns = [item for item in naive_patterns if item['stock'] == s]
    max_cuts_dict = {f: max([filtered_patterns[i]['cuts'][f] for i,_ in enumerate(filtered_patterns)]) for f in finish.keys()}
    min_cuts_dict = {f: min([filtered_patterns[i]['cuts'][f] for i,_ in enumerate(filtered_patterns)]) for f in finish.keys()}

    return min_cuts_dict, max_cuts_dict

def generate_cut_combinations(stock_id, min_c_values, max_c_values, pattern):
    """
    This function generates combinations of 'cuts'- Finish Goods ID with their respective 'C' Finished Goods values,
    based on the provided minimum and maximum 'C' values for each cut.

    Parameters:
        stock_id (int or str): Identifier for the stock.
        min_c_values (dict): Dictionary containing the minimum 'C' values for each FG.
        max_c_values (dict): Dictionary containing the maximum 'C' values for each FG.
        pattern (list): List to store the generated cut combinations.

    Returns:
        None

    Description:
        This function generates all possible combinations of 'cuts' along with their respective 'F' values
        within the given ranges. It uses a recursive approach to iterate through all possible combinations
        of 'F' values for each 'cut'. The generated combinations are appended to the 'pattern' list in the
        format {'stock': stock_id, 'cuts': current_combination}, where 'current_combination' is a dictionary
        representing a combination of 'cuts' and their corresponding 'F' values.
    """
    def generate_combinations_util(keys, current_combination):
        """
        Utility function to generate combinations recursively.

        Parameters:
            keys (list): List of remaining keys (Finish Goods ID) to be considered.
            current_combination (dict): Current combination of 'Finish Goods ID' and their 'C' values.

        Returns:
            None
        
        Improve: Tao combination tu cao xuong thap, va uu tien stock co width lon)- Done
        """
        if not keys:
            if any(current_combination.values()):  # Check if any 'F' value is non-zero, remove the case all zeros
                pattern.append({'stock': stock_id, 'cuts': current_combination})
            return

        key = keys[0]
        min_c = min_c_values.get(key, 0)
        max_c = max_c_values.get(key, 0)

        for c in range(max_c, min_c - 1, -1):
            new_combination = current_combination.copy()
            new_combination[key] = c
            generate_combinations_util(keys[1:], new_combination)

    reorderd_max_c = change_order_dict(max_c_values)
    keys = list(reorderd_max_c.keys())
    generate_combinations_util(list(keys), {})
    return pattern


# ADD TIME OUT FOR FUNCTION WITH TOO LARGE COMBINATION
def generate_cut_combinations_with_timeout(stock_id, min_c_values, max_c_values, pattern, TIMEOUT):

    def generate_combinations_util(keys, current_combination, start_time, TIMEOUT):

        if not keys:
            if any(current_combination.values()):  # Check if any 'F' value is non-zero, remove the case all zeros
                pattern.append({'stock': stock_id, 'cuts': current_combination})
            return
        
        key = keys[0]
        min_c = min_c_values.get(key, 0) #retrieves the value associated, 0 if not found
        max_c = max_c_values.get(key, 0)

        for c in range(max_c, min_c - 1, -1):
            # Check if the timeout has been exceeded
            if time.time() - start_time >= TIMEOUT:
                return pattern  # Return the pattern if timeout occurs
            else:
                new_combination = current_combination.copy()
                new_combination[key] = c
                generate_combinations_util(keys[1:], new_combination, start_time, TIMEOUT)

    reorderd_max_c = change_order_dict(max_c_values)
    keys = list(reorderd_max_c.keys())

    import time
    start_time = time.time()
    generate_combinations_util(list(keys), {},start_time, TIMEOUT)