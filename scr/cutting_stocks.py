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
    stocks = {'TP235H002656-2': {'receiving_date': 45150, 'width': 1219, 'weight': 4332.5}, 
            'TP235H002656-1': {'receiving_date': 44951, 'width': 1219, 'weight': 4332.5},
             'TP232H001074': {'receiving_date': 45040, 'width': 1233, 'weight': 7105.0},
             'TP232H001073': {'receiving_date': 45153, 'width': 1233, 'weight': 7550.0}, 
            'TP236H005198': {'receiving_date': 45011, 'width': 1136, 'weight': 8000.0},
             'TP235H002652': {'receiving_date': 45039, 'width': 1219, 'weight': 8400.0}, 
            'TP235H002654': {'receiving_date': 45045, 'width': 1219, 'weight': 8500.0}, 
            'TP232H001072': {'receiving_date': 45013, 'width': 1233, 'weight': 8675.0}, 
            'TP235H002655': {'receiving_date': 45229, 'width': 1219, 'weight': 8845.0}, 
            'TP235H002653': {'receiving_date': 45045, 'width': 1219, 'weight': 8855.0}, 
            'TP232H001075': {'receiving_date': 45247, 'width': 1233, 'weight': 9630.0}}

    finish = {'F23': {'width': 306.0,  'need_cut': 839.0,  'upper_bound': 1548.5841833599998, 
                      'fc1': 2365.2806112,  'fc2': 3692.5657704,  'fc3': 3457.8613836}, 
            'F22': {'width': 205.0,  'need_cut': 498.79081,  'upper_bound': 3362.2258921410994, 
                    'fc1': 9544.7836,  'fc2': 5494.6232,  'fc3': 3908.5464},
            'F21': {'width': 188.0,  'need_cut': 30772.5997,  'upper_bound': 39966.4243228516,  
                     'fc1': 30646.0820436,  'fc2': 35762.3146452,  'fc3': 34039.2591132},
            'F20': {'width': 175.0,  'need_cut': 28574.78588,  'upper_bound': 36618.447115077855, 
                     'fc1': 26812.20409,  'fc2': 31288.38713,  'fc3': 29780.88883}, 
            'F19': {'width': 155.0,  'need_cut': 4401.84053,  'upper_bound': 5851.570175548759, 
                    'fc1': 4832.4321325,  'fc2': 5639.1860525,  'fc3': 5367.4857775},
            'F18': {'width': 133.0,  'need_cut': 400.0,  'upper_bound': 795.8254562499999, 
                     'fc1': 1319.4181875,  'fc2': 759.546375,  'fc3': 540.295875}, 
            'F17': {'width': 120.0,  'need_cut': 1751.0,  'upper_bound': 2526.6533504, 
                    'fc1': 2585.511168,  'fc2': 4319.793456,  'fc3': 3797.778504}, 
            'F24': {'width': 82.0,  'need_cut': 977.9362,  'upper_bound': 1585.531389098001, 
                    'fc1': 2025.3170816,  'fc2': 3383.8382072,  'fc3': 2974.9264948}}

    return stocks, finish

def calculate_upper_bounds(finish): # FIX BY THE OPERATOR AND THE BOUND calculate upper_bound according to the (remained) need_cut and
    return {f: {**f_info, "upper_bound": f_info['need_cut'] + f_info['fc1']} for f, f_info in finish.items()}

def cutting_stocks(MIN_MARGIN, dual_stocks, dual_finish, stocks,finish, final_solution_patterns):
    n = 0
    rm_stock = True
    max_key = None
    # print("PHASE 1: Naive/ Dual Pattern Generation",end=".")    
    patterns = make_naive_patterns(dual_stocks,dual_finish,MIN_MARGIN) # FIX ->> neu ko co pattern nao phu hop thi gian UPPERBOUND
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
    taken_stocks = {p['stock'] for p in final_solution_patterns} 

    for f, f_info in dual_finish.items():
        if over_cut[f] <0:
            f_info['need_cut'] = -over_cut[f]
        else: 
            f_info['need_cut'] = 0
            f_info['upper_bound'] += -over_cut[f]
        
    # remained_finish = {
    #     f: {**f_info, 'need_cut': -over_cut[f]}
    #     for f, f_info in dual_finish.items()
    #     if over_cut[f] < 0                      # continue to cut if negative cut
    # }
    
    # Prepare remained_stocks dictionary
    remained_stocks = {
        s: {**s_info}
        for s, s_info in dual_stocks.items()
        if s not in taken_stocks
    }
    return dual_finish, remained_stocks

def check_conditions(overused_list,remained_stocks):
    # CHECK FOR ANOTHER ROUND
    if not overused_list or not remained_stocks:
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
        logger.info(f'overcut amount {over_cut}')
        remained_finish, remained_stocks = refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut)
        cond = check_conditions(overused_list, remained_stocks)
        if not cond:
            logger.info(f"Total stock used {len(final_solution_patterns)}")
            logger.info(f'TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}')
            # print([p['stock'] for p in final_solution_patterns])
            over_cut_rate = {key: round(over_cut[key]/finish[key]['fc1'], 4) for key in over_cut.keys()}
            logger.info(f">> TOTAL STOCK OVER CUT: {over_cut_rate}")
        else:
            # print(f">> TOTAL STOCK OVER CUT: {over_cut}") SUA LAI PHAN PICK NEXT STOCK, NHUNG STOCK CO THE TIEP TUC CAT DU THI NEN BO VAO THU
            dual_stocks = copy.deepcopy(remained_stocks)
            dual_finish = copy.deepcopy(remained_finish)
            # dual_finish = calculate_upper_bounds(remained_finish)
    
if __name__ =="__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'cutting_stocks{datetime.date.today()}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    batch = "POSCO+SAPH440-PO+4"
    logger.info(f'Started {batch}')
    logger.info('add contraint upperbound')
    logger.info(f'Process PARAMS')
    loop_cutting()
    logger.info('Finished')

    
