import pandas as pd
import numpy as np
import math

## PARAMETER

# over cut bound
BOUND = 0.3
MIN_MARGIN = 8


def make_patterns_by_weight_width(stocks, finish, BOUND, MIN_MARGIN):
    """
    Generates patterns of feasible cuts from stock lengths to meet specified finish lengths.

    Parameters:
    stocks (dict): A dictionary where keys are stock identifiers and values are dictionaries
                   with key 'length' representing the length of each stock.

    finish (dict): A dictionary where keys are finish identifiers and values are dictionaries
                   with key 'length' representing the required finish lengths.

    Returns:
    patterns (list): A list of dictionaries, where each dictionary represents a pattern of cuts.
                   Each pattern dictionary contains 'stock' (the stock identifier) and 'cuts'
                   (a dictionary where keys are finish identifiers and the value is the number
                   of cuts from the stock for each finish).
                   

                   Naive pattern with maximum number of cuts that is closet to the required need_cut
                   and SUM(length) smaller Mother Coil length
    """
    if BOUND is None:
        BOUND = 0.3
    else:
        pass

    if MIN_MARGIN is None:
        MIN_MARGIN == 8
    else:
        pass

    patterns = []
    for f in finish:
        feasible = False
        # finish[f]["upper_cut"] = finish[f]["need_cut"] + 

        for s in stocks:
            # max number of f that fit on s 
            num_cuts_by_width = int((stocks[s]["width"]-MIN_MARGIN) / finish[f]["width"])
            # max number of f that satisfied the need cut
            upper_demand_finish = finish[f]["need_cut"] + BOUND * finish[f]["need_cut"]
            num_cuts_by_weight = int((upper_demand_finish * stocks[s]["width"] ) / (finish[f]["width"]*stocks[s]['weight']))
            # min of two max will satisfies both
            num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

            # make pattern and add to list of patterns
            if num_cuts > 0:
                feasible = True
                cuts_dict = {key: 0 for key in finish.keys()}
                cuts_dict[f] = num_cuts
                patterns.append({"stock": s, "cuts": cuts_dict})
        if not feasible:
            print(f"No feasible pattern was found for Stock {s} and FG {f}")
            # return []

    return patterns

def ap_upper_bound(naive_patterns,finish):
    #upper bound of possible cut over all kind of stocks
    a_upper_bound = { 
    f: max([naive_patterns[i]['cuts'][f] for i,_ in enumerate(naive_patterns)])
    for f in finish.keys()
    }
    return a_upper_bound