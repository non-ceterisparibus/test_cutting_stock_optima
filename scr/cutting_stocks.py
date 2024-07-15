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
            ,"spec_name": "SAPH440-PO" # yeu cau chuan hoa du lieu OP - PO
            ,"thickness": 4
            ,"maker" : "POSCO"
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
    stocks = {'TR246H00000085': {'receiving_date': 45170, 'width': 1219, 'weight': 6820.0},
              'TM246H00000163': {'receiving_date': 45041, 'width': 1219, 'weight': 6860.0},
             'TP235H002665': {'receiving_date': 45229, 'width': 1219, 'weight': 7095.0},
             'TP235H002666': {'receiving_date': 44956, 'width': 1219, 'weight': 7170.0}, 
            'TP234H001880': {'receiving_date': 45037, 'width': 1219, 'weight': 8535.0},
             'TM246H00000161': {'receiving_date': 45039, 'width': 1219, 'weight': 8610.0}}

    finish = {'F28': {'width': 225.0,  'need_cut': 3442.0,  'upper_bound': 5246.0615974, 
         'fc1': 6013.538658,  'fc2': 9388.0560735,  'fc3': 8791.33875525}, 
              'F27': {'width': 150.0,  'need_cut': 663.0,  'upper_bound': 1691.686608, 
        'fc1': 3428.95536,  'fc2': 5728.99437,  'fc3': 5036.687955}}

    return stocks, finish

def calculate_upper_bounds(finish):
    # Re calculate upper_bound according to the (remained) need_cut and
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
    taken_stocks = {p['stock'] for p in final_solution_patterns}  # Using a set for faster lookups
    # Prepare finish_cont dictionary
    
    # remained_finish = {}
    for f, f_info in dual_finish.items():
        if over_cut[f] <0:
            f_info['need_cut'] = -over_cut[f]
        else: 
            f_info['need_cut'] = 0
            f_info['upper_bound'] = f_info['fc1'] -over_cut[f]
        
    # remained_finish = {
    #     f: {**f_info, 'need_cut': -over_cut[f]}
    #     for f, f_info in dual_finish.items()
    #     if over_cut[f] < 0                      # continue to cut if negative cut
    # }
    
    # Prepare stocks_cont dictionary
    remained_stocks = {
        s: {**s_info}
        for s, s_info in dual_stocks.items()
        if s not in taken_stocks
    }
    return dual_finish, remained_stocks

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
            over_cut_rate = {key: round(over_cut[key]/finish[key]['fc1'], 4) for key in over_cut.keys()}
            logger.info(f">> TOTAL STOCK OVER CUT: {over_cut_rate}")
        else:
            # print(f">> TOTAL STOCK OVER CUT: {over_cut}") SUA LAI PHAN PICK NEXT STOCK, NHUNG STOCK CO THE TIEP TUC CAT DU THI NEN BO VAO THU
            remained_finish, remained_stocks = refresh_data(final_solution_patterns, dual_finish, dual_stocks, over_cut)
            dual_stocks = copy.deepcopy(remained_stocks)
            dual_finish = calculate_upper_bounds(remained_finish)
    
if __name__ =="__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'cutting_stocks{datetime.date.today()}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    batch = "POSCO+SAPH440-PO+4"
    logger.info(f'Started {batch}')
    logger.info(f'Process PARAMS')
    loop_cutting()
    logger.info('Finished')

    
