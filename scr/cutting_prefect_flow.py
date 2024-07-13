from prefect import task, flow
import random
import time
from typing import Dict, Any
import copy
import json

from auxilaries import *
from solvers import *

@task
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
     'TP232H001075': {'receiving_date': 45247, 'width': 1233, 'weight': 9630.0}
    }

    finish = {'F23': {'width': 306.0,  'need_cut': 839.0,  'upper_bound': 1548.5841833599998,
      'fc1': 2365.2806112,  'fc2': 3692.5657704,  'fc3': 3457.8613836},
     'F22': {'width': 205.0,  'need_cut': 498.7908121410992,  'upper_bound': 3362.2258921410994,
      'fc1': 9544.7836,  'fc2': 5494.6232,  'fc3': 3908.5464},
     'F21': {'width': 188.0,  'need_cut': 30772.599709771595,  'upper_bound': 39966.4243228516,
      'fc1': 30646.0820436,  'fc2': 35762.3146452,  'fc3': 34039.2591132},
     'F20': {'width': 175.0,  'need_cut': 28574.78588807786,  'upper_bound': 36618.447115077855,
      'fc1': 26812.20409,  'fc2': 31288.38713,  'fc3': 29780.88883},
     'F19': {'width': 155.0,  'need_cut': 4401.8405357987585,  'upper_bound': 5851.570175548759,
      'fc1': 4832.4321325,  'fc2': 5639.1860525,  'fc3': 5367.4857775},
     'F18': {'width': 133.0,  'need_cut': 400.0,  'upper_bound': 795.8254562499999,
      'fc1': 1319.4181875,  'fc2': 759.546375,  'fc3': 540.295875},
     'F17': {'width': 120.0,  'need_cut': 1751.0,  'upper_bound': 2526.6533504,
      'fc1': 2585.511168,  'fc2': 4319.793456,  'fc3': 3797.778504},
     'F24': {'width': 82.0,  'need_cut': 977.9362646180011,  'upper_bound': 1585.531389098001,
      'fc1': 2025.3170816,  'fc2': 3383.8382072,  'fc3': 2974.9264948}
    }
    return stocks, finish

@task
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

    with open('margin_dict.json', 'r') as file:
        margin_dict = json.load(file)

    # GET ALL PARAMS
    MIN_MARGIN = margin_dict[PARAMS["warehouse"]][f"thickness_{PARAMS['thickness']}"]["margin"]
    print(f"MIN_MARGIN:{MIN_MARGIN}")

    # BOUND_KEY = next(iter(PARAMS['stock_ratio']))
    # BOUND_VALUE = PARAMS['stock_ratio'][BOUND_KEY]
    # print(f"BOUND_VALUE:{BOUND_VALUE}")
    return MIN_MARGIN

@task
def cutting_stocks(MIN_MARGIN, total_dual_pat,dual_stocks, dual_finish, nai_patterns, stocks,finish, final_solution_patterns):
    i = 0
    rm_stock = True
    max_key = None
    while rm_stock == True:
        dual_stocks = filter_out_stock_by_cond(dual_stocks, max_key)
        len_stocks = len(dual_stocks)
        # print("PHASE 1: Naive/ Dual Pattern Generation",end=".")    
        patterns = make_naive_patterns(dual_stocks, dual_finish, MIN_MARGIN)
        dual_finish = create_finish_demand_by_line_fr_naive_pattern(patterns, dual_finish)

        # print("PHASE 2: Pattern Duality",end=".")
        new_pattern = generate_pattern_dual(dual_stocks, dual_finish, patterns, MIN_MARGIN) # Stock nao do toi uu hon stock khac o width thi new pattern luon bi chon cho stock do #FIX
        dual_pat = []
        while new_pattern not in dual_pat:
            patterns.append(new_pattern)        # pattern de generate them new pattern
            total_dual_pat.append(new_pattern)  # tinh tong dual pattern co the duoc generate
            dual_pat.append(new_pattern)        # dual pat de tinh stock bi lap nhieu lan
            new_pattern = generate_pattern_dual(dual_stocks, dual_finish, patterns, MIN_MARGIN)
            print(end=".")

        # filter stock having too many patterns
        ls = count_pattern(dual_pat)
        max_key = max(ls, key=ls.get) # take the max 
        max_count = ls[max_key]
        if max_count > 1 and i < len_stocks - 2:# con lai it nhat 2 stock
            rm_stock = True
            i +=1
            print(f"{i} round")
        else: 
            rm_stock = False

    # Phase 1.2: Combine pattterns
    sum_patterns = nai_patterns + total_dual_pat

    # Phrase: Filter Patterns having trim loss as requirements
    # print("PHASE 3: Filter Patterns", end=".")
    filtered_trimloss_pattern = []
    idx=0
    for pattern in sum_patterns:
        cuts_dict= pattern['cuts']
        width_s = stocks[pattern['stock']]['width']
        trim_loss = width_s - sum([finish[f]["width"] * cuts_dict[f] for f in cuts_dict.keys()])
        trim_loss_pct = round(trim_loss/width_s * 100, 3)
        if trim_loss_pct <= 4.00: # filter for naive pattern
            idx +=1
            pattern.update({'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct, "patt_id":idx})
            filtered_trimloss_pattern.append(pattern)

    # print("PHASE 4: Cut Patterns", end=".")
    filtered_stocks = [filtered_trimloss_pattern[i]['stock'] for i in range(len(filtered_trimloss_pattern))]
    chosen_stocks = {}
    for stock_name, stock_info in stocks.items():
        if stock_name in filtered_stocks:
            chosen_stocks[stock_name]= {**stock_info}

    solution, total_cost = cut_weight_patterns(chosen_stocks, dual_finish, filtered_trimloss_pattern)
    solution_list = []
    # print(f"Total stock need: {total_cost}")
    for i, pattern_info in enumerate(filtered_trimloss_pattern):
        count = solution[i]
        if count > 0:
            s = pattern_info['stock']
            sol = {"count": count, **pattern_info}
            solution_list.append(sol)
    
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

    over_cut = {}
    over_cut_ratio = {}
    # Overcut cutweight (slice * fg_width * wu) - need_cut 
    for key in total_sums.keys():
        over_cut[key] = round(total_sums[key] - finish[key]['need_cut'],3)
        over_cut_ratio[key] = round(over_cut[key]/finish[key]['fc1'], 4)
        
    return final_solution_patterns, over_cut, over_cut_ratio,overused_list

@task
def refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut):
    # Extract stocks from final_solution_patterns
    taken_stocks = {p['stock'] for p in final_solution_patterns}  # Using a set for faster lookups
    # Prepare finish_cont dictionary
    finish_cont = {
        f: {**f_info, 'need_cut': -over_cut[f]}
        for f, f_info in dual_finish.items()
        if over_cut[f] < 0
    }
    # Prepare stocks_cont dictionary
    stocks_cont = {
        s: {**s_info}
        for s, s_info in dual_stocks.items()
        if s not in taken_stocks
    }
    return finish_cont, stocks_cont

@task
def check_condition(overused_list):
    # CHECK FOR ANOTHER ROUND
    if not overused_list:
        print("\n FINISH CUTTING")
        return False
    else:
        # go back to PHRASE 3
        print("\n BACK TO PHRASE 3: TO FILTER OUT PATTERNS")
        return True
    
@flow
def loop_cutting():
    MIN_MARGIN = load_params()
    stocks, finish = load_data()
    print("SETUP")
    final_solution_patterns =[]

    # CAN BE RESET AT EACH ROUND
    dual_stocks = copy.deepcopy(stocks)
    dual_finish = copy.deepcopy(finish)
    # nai_patterns = make_naive_patterns(stocks, finish, MIN_MARGIN)
    cond = True
    
    while cond == True:
        total_dual_pat = []
        nai_patterns = make_naive_patterns(dual_stocks, dual_finish, MIN_MARGIN)
        
        # START LOOP
        final_solution_patterns, over_cut, over_cut_ratio, overused_list = cutting_stocks(MIN_MARGIN, total_dual_pat, dual_stocks, dual_finish, nai_patterns, stocks,finish, final_solution_patterns)
        cond = check_condition(overused_list)
        if not cond:
            print([p['stock'] for p in final_solution_patterns])
            # print(f"Total stock used:{len(final_solution_patterns)}")
            print(f">> Stock-ratio on next month forecast: {over_cut_ratio}")
            print(f"no stock used {len(final_solution_patterns)}")
        else:
            finish_cont, stocks_cont = refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut)
            dual_stocks = copy.deepcopy(stocks_cont)
            dual_finish = copy.deepcopy(finish_cont)
    
if __name__ =="__main__":
    loop_cutting()
    
