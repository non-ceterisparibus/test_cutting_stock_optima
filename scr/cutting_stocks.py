import random
import datetime 
from typing import Dict, Any
import copy
import json
import logging

from auxilaries import *
from solvers import *

def load_params():
    PARAMS = {"warehouse": "HSC"
          ,"spec_name": "JSH590R-PO" # yeu cau chuan hoa du lieu OP - PO
          ,"thickness": 2
          ,"maker" : "CSC"
          ,"stock_ratio": { #get from app 
                    "limited": None
                    # "default": 2          ->>>> OPERATOR FLOW
                    # "user_setting": 4
                }
        #   ,"forecast_scenario": median
          }

    with open('scr/margin_dict.json', 'r') as file:
        margin_dict = json.load(file)

    # GET ALL PARAMS
    MIN_MARGIN = margin_dict[PARAMS["warehouse"]][f"thickness_{PARAMS['thickness']}"]["margin"]
    return MIN_MARGIN

def load_data():
    stocks = {'HTV1983/23-2': {'receiving_date': 45142, 'width': 1219, 'weight': 5485.0},
    'HTV1983/23-1': {'receiving_date': 45017, 'width': 1219, 'weight': 5485.0},
    'HTV1060/24': {'receiving_date': 45135, 'width': 1219, 'weight': 11295.0},
    'HTV1059/24': {'receiving_date': 45149, 'width': 1219, 'weight': 11470.0}}

    finish = {'F147': {'width': 180.0,  'need_cut': 1900.0,  'upper_bound': 2639.22928,
            'fc1': 2464.0976,  'fc2': 2991.6864,  'fc3': 3116.0216},
            'F146': {'width': 135.0,  'need_cut': 148.0,  'upper_bound': 591.5031999999999,
             'fc1': 1478.3439999999998,  'fc2': 1047.788,  'fc3': 860.8880000000001},
            'F145': {'width': 130.0,  'need_cut': 1300.0,  'upper_bound': 1614.1936, 
            'fc1': 1047.3120000000001,  'fc2': 1127.448,  'fc3': 1355.76},
            'F144': {'width': 125.0,  'need_cut': 857.0,  'upper_bound': 1649.82368,
             'fc1': 2642.7456,  'fc2': 1723.6512000000002,  'fc3': 1700.3712000000003},
            'F143': {'width': 104.0,  'need_cut': 994.0,  'upper_bound': 2488.4296,
             'fc1': 4981.432,  'fc2': 4540.3060000000005,  'fc3': 5178.620000000001},
            'F152': {'width': 90.0,  'need_cut': 4778.0,  'upper_bound': 8071.152320000001,
             'fc1': 10977.174400000002,  'fc2': 9940.747600000002,  'fc3': 11556.632},
            'F151': {'width': 75.0,  'need_cut': 500.0,  'upper_bound': 711.5,
            'fc1': 705.0,  'fc2': 628.5,  'fc3': 726.0},
            'F150': {'width': 72.0,  'need_cut': 5015.0,  'upper_bound': 6995.11172,
             'fc1': 6600.372399999999,  'fc2': 5905.539400000001,  'fc3': 7208.1378},
            'F149': {'width': 58.0,  'need_cut': 700.0,  'upper_bound': 866.3199999999999,
            'fc1': 554.4,  'fc2': 449.99999999999994,  'fc3': 601.1999999999999},
            'F148': {'width': 47.0,  'need_cut': 300.0,  'upper_bound': 598.89,
            'fc1': 996.3,  'fc2': 900.3399999999999,  'fc3': 1037.8299999999997}}
    
    return stocks, finish

def calculate_upper_bounds(finish):
    # Re calculate upper_bound according to the (remained) need_cut and
    for key, finish_info in finish.items():
        finish[key] = {**finish_info, "upper_bound": finish_info['need_cut'] + finish_info['fc1']}
    return finish

def cutting_stocks(MIN_MARGIN, dual_stocks, dual_finish, stocks,finish, final_solution_patterns):
    n = 0
    rm_stock = True
    max_key = None
    # print("PHASE 1: Naive/ Dual Pattern Generation",end=".")    
    patterns = make_naive_patterns(dual_stocks,dual_finish,MIN_MARGIN)
    dual_finish = create_finish_demand_by_line_fr_naive_pattern(patterns, dual_finish)
    len_stocks = len(dual_stocks)
    while rm_stock == True:
        # print("PHASE 2: Pattern Duality",end=".")
        dual_stocks = filter_out_stock_by_cond(dual_stocks, max_key)
        new_pattern = generate_pattern_dual(dual_stocks, dual_finish, patterns, MIN_MARGIN) # Stock nao do toi uu hon stock khac o width thi new pattern luon bi chon cho stock do #FIX
        dual_pat = []
        while new_pattern not in dual_pat:
            patterns.append(new_pattern)        # pattern de generate them new pattern
            dual_pat.append(new_pattern)        # dual pat de tinh stock bi lap nhieu lan
            new_pattern = generate_pattern_dual(dual_stocks, dual_finish, patterns, MIN_MARGIN)
            print(end=".")

        # filter stock having too many patterns
        ls = count_pattern(dual_pat)
        max_key = max(ls, key=ls.get) # take the max 
        max_count = ls[max_key]
        if max_count > 1 and n < len_stocks - 2:
            rm_stock = True
            n +=1
            print(f"{n} round")
        else: 
            rm_stock = False

    # print("PHASE 3: Filter Patterns", end=".")
    filtered_trimloss_pattern = []
    for pattern in patterns:
        cuts_dict= pattern['cuts']
        width_s = stocks[pattern['stock']]['width']
        trim_loss = width_s - sum([finish[f]["width"] * cuts_dict[f] for f in cuts_dict.keys()])
        trim_loss_pct = round(trim_loss/width_s * 100, 3)
        if trim_loss_pct <= 4.00: # filter for naive pattern
            pattern.update({'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct})
            filtered_trimloss_pattern.append(pattern)

    # print("PHASE 4: Cut Patterns", end=".")
    filtered_stocks = [filtered_trimloss_pattern[i]['stock'] for i in range(len(filtered_trimloss_pattern))]
    chosen_stocks = {}
    for stock_name, stock_info in stocks.items():
        if stock_name in filtered_stocks:
            chosen_stocks[stock_name]= {**stock_info}

    _, solution_list = cut_weight_patterns(chosen_stocks, dual_finish, filtered_trimloss_pattern)
    # print(f'Solution list {[p['stock'] for p in solution_list]}')
    
    # Neu lap stock thi rm all pattern tru cai trim loss thap nhat va chay lai
    sorted_solution_list = sorted(solution_list, key=lambda x: (x['stock'],  x.get('trim_loss_pct', float('inf'))))

    # now take first overused stock pattern only.
    overused_list = []
    take_stock = None
    for solution_pattern in sorted_solution_list:
        current_stock = solution_pattern['stock']
        if current_stock == take_stock:
            overused_list.append(solution_pattern)
        else:
            take_stock = current_stock
            final_solution_patterns.append(solution_pattern)
                
    # Phase 5: Evaluation Over-cut / Stock Ratio
    print("\n PHASE 5: Evaluation Stock Ratio", end=".")
    for i, sol in enumerate(final_solution_patterns):
        s = final_solution_patterns[i]['stock'] # stock cut
        cuts_dict = final_solution_patterns[i]['cuts']
        weight_dict = {f: round(cuts_dict[f] * finish[f]['width']*stocks[s]['weight']/stocks[s]['width'],3) for f in cuts_dict.keys()}
        final_solution_patterns[i] = {**sol, "cut_w": weight_dict}
    
    # Total Cuts
    total_sums = count_weight(final_solution_patterns)
    over_cut = {k: round(total_sums[k] - finish[k]['need_cut'],3) for k in total_sums.keys() }
    
    return final_solution_patterns, over_cut, overused_list

def refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut):
    # Extract stocks from final_solution_patterns
    taken_stocks = {p['stock'] for p in final_solution_patterns}  # Using a set for faster lookups
    # Prepare finish_cont dictionary
    remained_finish = {
        f: {**f_info, 'need_cut': -over_cut[f]}
        for f, f_info in dual_finish.items()
        if over_cut[f] < 0                      # continue to cut if negative cut
    }
    # Prepare stocks_cont dictionary
    remained_stocks = {
        s: {**s_info}
        for s, s_info in dual_stocks.items()
        if s not in taken_stocks
    }
    return remained_finish, remained_stocks

def check_conditions(overused_list):
    
    # CHECK FOR ANOTHER ROUND
    if not overused_list:
        print("\n FINISH CUTTING")
        return False
    else:
        # go back to PHRASE 3
        print("\n BACK TO PHRASE 3: TO FILTER OUT PATTERNS")
        return True

def loop_cutting():
    # print("LOAD DATA/PARAMS")
    MIN_MARGIN = load_params()
    logger.info(f"MIN_MARGIN:{MIN_MARGIN}")
    stocks, finish = load_data()
    final_solution_patterns =[]
    
    print("SETUP STOCK - FINISHED GOOD")
    # CAN BE RESET AT EACH ROUND
    dual_stocks = copy.deepcopy(stocks)
    dual_finish = calculate_upper_bounds(finish)
    cond = True
    while cond == True:
        # START LOOP
        final_solution_patterns, over_cut, overused_list = cutting_stocks(MIN_MARGIN, dual_stocks, dual_finish, stocks,finish, final_solution_patterns)
        logger.info(f'Take stock {[p['stock'] for p in final_solution_patterns]}')
        cond = check_conditions(overused_list)
        if not cond:
            logger.info(f"Total stock used {len(final_solution_patterns)}")
            logger.info(f'TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}')
            # print([p['stock'] for p in final_solution_patterns])
            over_cut_rate = [key: round(over_cut[key]/finish[key]['fc1'], 4) for key in finish.keys()]
            logger.info(f">> TOTAL STOCK OVER CUT: {over_cut_rate}")
        else:
            # print(f">> TOTAL STOCK OVER CUT: {over_cut}") SUA LAI PHAN PICK NEXT STOCK, NHUNG STOCK CO THE TIEP TUC CAT DU THI NEN BO VAO THU
            remained_finish, remained_stocks = refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut)
            dual_stocks = copy.deepcopy(remained_stocks)
            dual_finish = calculate_upper_bounds(remained_finish)
    
    
if __name__ =="__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'cutting_stocks{datetime.date.today()}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('Started')
    logger.info(f'Process PARAMS')
    loop_cutting()
    logger.info('Finished')

    
