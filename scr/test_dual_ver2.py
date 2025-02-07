# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import numpy as np
import traceback
import logging
import datetime
import time
import json
import copy

from model import CuttingStocks
from model import SemiProb
import re
import os

# LOAD ENV PARAMETERS & DATA
global max_bound
global no_warehouse
global mc_ratio
global stock_ratio
global added_stock_ratio_avg_fc
global customer_group
global max_coil_weight
global bound_step
global starting_bound
global stop_stock_ratio
global stop_needcut_wg


max_coil_weight         = float(os.getenv('MAX_WEIGHT_MC_DIV_2', '7000'))
max_bound               = float(os.getenv('MAX_BOUND', '3.0'))
no_warehouse            = float(os.getenv('NO_WAREHOUSE', '3.0'))
added_stock_ratio_avg_fc = [3000, 1500, 500]
bound_step = 0.5

# # User adjustably  
# mc_ratio        = float(os.getenv('MC_RATIO', '2.5'))
stop_stock_ratio = float(os.getenv('ST_RATIO', '-0.03'))
stop_needcut_wg = -90

# HELPERS
def create_warehouse_order(finish):
    finish_dict_value = dict(list(finish.values())[0])
    coil_center_1st = finish_dict_value['1st Priority']
    if finish_dict_value['2nd Priority'] == 'x':
        coil_center_2nd = None
    else:
        coil_center_2nd = finish_dict_value['2nd Priority']
    
    if finish_dict_value['3rd Priority'] == 'x':
        coil_center_3rd = None
    else:
        coil_center_3rd = finish_dict_value['3rd Priority']
    
    warehouse_order = [coil_center_1st,coil_center_2nd,coil_center_3rd]
    return warehouse_order

def filter_stocks_by_wh(stock, warehouse_order):
    # Filter and order the dictionary
    return {k: v for w in warehouse_order for k, v in stock.items() if v['warehouse'] == w}

def partite_stock(stocks, allowed_cut_weight, group, medium_mc_weight):
    """Take amount of stock equals 2.5 to 3.0 need cut
    Args:
        stocks (dict): _description_
        allowed_cut_weight (float): _description_

    Returns:
        partial_stocks: dict
    """
    max_mc_wg = max([v['weight'] for _,v in stocks.items()])
    
    if "big" in group:
        filtered_stocks = {k: v for k, v in stocks.items() if v['weight'] >= max_coil_weight}
        # Sort weight ascending and receiving_date ascending
        filtered_stocks = dict(sorted(filtered_stocks.items(), key=lambda x: ( x[1]['receiving_date'],x[1]['weight'])))
    elif (group in ["medium","medium1","medium2",'medium3'] and medium_mc_weight*2 > max_mc_wg):
        # Sort weight ascending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (x[1]['receiving_date'],x[1]['weight'])))
    elif (group in ["medium","medium1","medium2",'medium3'] and medium_mc_weight*2 <= max_mc_wg):
        # Sort weight descending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (-x[1]['receiving_date'],x[1]['weight']), reverse= True))
    else: 
        # Sort weight descending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (-x[1]['receiving_date'],x[1]['weight']), reverse= True))
    
    partial_stocks = {}
    accumulated_weight = 0
    
    for s, sinfo in filtered_stocks.items():
        accumulated_weight += sinfo['weight']
        if allowed_cut_weight * 1.2 <= accumulated_weight:
            if len(partial_stocks) <= 1:
                partial_stocks[s] = {**sinfo}
            else:
                break
        else:
            partial_stocks[s] = {**sinfo}
    res = dict(sorted(partial_stocks.items(), key=lambda x: (x[1]['weight']), reverse=True))
 
    return res

def partite_finish(finish, stock_ratio):
    """_Select more FG codes (finish) below the indicated stock ratio to reduce trim loss _

    Args:
        finish (_type_): _description_
        stock_ratio (_type_): _description_

    Returns:
        partial_finish: the proportion of finish has the stock ratio as required
    """
    partial_pos_finish = {}
    for avg_fc in added_stock_ratio_avg_fc:
        print(f"range forecast {avg_fc}")
        for f, finfo in finish.items():
            average_fc = max(finfo.get('average FC', 0), 1)
            fg_ratio = finfo['need_cut'] / average_fc
            if (0 <= fg_ratio <= stock_ratio and round(finfo['average FC']) >= avg_fc):
                # stock ratio > 0
                partial_pos_finish[f] = finfo.copy()
        
        if len(partial_pos_finish) >= 1:
            def_avg_fc = avg_fc
            break
        else:
            continue
    
    partial_finish = {}
    for f, finfo in finish.items():
        # Ensure 'average FC' is treated as 1 if it's 0
        average_fc = max(finfo.get('average FC', 0), 1)
        fg_ratio = finfo['need_cut'] / average_fc
        
        # Safely retrieve values with default fallback
        def_avg_fc = def_avg_fc if 'def_avg_fc' in locals() else 500
    
        # Check conditions for partial finishes
        if (
            fg_ratio < 0 
            or (0 <= fg_ratio <= stock_ratio and round(average_fc) >= def_avg_fc)
        ):
            partial_finish[f] = finfo.copy()

    res = dict(sorted(partial_finish.items(), key=lambda x: (x[1]['need_cut'],x[1]['width']))) # need_cut van dang am sort ascending               
    
    return res

def refresh_finish(finish, over_cut):
    # Update need cut
    for f in over_cut.keys(): # neu f khong co trong over_cut thi tuc la finish[f] chua duoc xu ly
        if over_cut[f] < 0:
            try: # finish stock ratio < -2% removed in previous run, still in overcut
                finish[f]['need_cut'] = over_cut[f] # finish need cut am
            except KeyError:
                pass    
        else:
            try: # finish removed in previous run wont appear in finish[f] but still in overcut
                finish[f]['need_cut'] = 0
                # finish[f]['upper_bound'] += -over_cut[f]
            except KeyError:
                pass
            
    # Take only finish with negative need_cut
    re_finish = {k: v for k, v in finish.items() 
                 if v['need_cut']/(v['average FC']+1) < stop_stock_ratio 
                 and v['need_cut']< stop_needcut_wg}
    if len(re_finish)== 0 or len(re_finish) > 3:
        pass # khong cat nua hoac co so FG ok
    elif len(re_finish) <=3 and sum(v['need_cut'] for _, v in finish.items())>= -200:
        re_finish = {} # Khong cat nua
    else: # them ma duong cat tiep
        for stop_rate in [0.0, 0.1, 0.3]:
            re_finish = {k: v for k, v in finish.items() if v['need_cut']/(v['average FC']+1) < stop_rate}
            if len(re_finish) >= 3:
                break
    return re_finish

def refresh_stocks(taken_stocks,stocks):
    if not taken_stocks:
        remained_stocks = stocks
    else:
        # if any taken stock have Div in the name pop -Div, from stock to use
        taken_og = []
        for s in taken_stocks:
            if str(s).__contains__("-Di"):
                og_s = str(s).split("-Di")[0]
                taken_og.append(og_s)
        taken_stocks = taken_stocks + taken_og
    
        # UPDATE stocks
        div_stock_list = list(set(taken_og))
        for stock_key in div_stock_list:
            try:
                half_wg = stocks[stock_key]['weight']*0.5
                for i in range(2):
                    stocks[f'{stock_key}-Di{i+1}'] = stocks[stock_key]
                    stocks[f'{stock_key}-Di{i+1}'].update({'weight': half_wg})
                    stocks[f'{stock_key}-Di{i+1}'].update({'status':"R:REWIND"})
            except KeyError: # already update in someround - the stock ID is the remained
                pass 
                 
        remained_stocks = {
                s: {**s_info}
                for s, s_info in stocks.items()
                if s not in taken_stocks
            }
    return remained_stocks

# Save File
def save_to_json(filename, data):
    with open(filename, 'w') as solution_file:
        json.dump(data, solution_file, indent=2)

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','count_cut','cuts','cut_width' ,"cut_weight", "remark","customer_short_name"]}
        for cut, line in item['cuts'].items():
            if line > 0:
                flattened_item = {**common_data, 
                                  'cuts': cut, 
                                  'lines': line,
                                  'fg_code':item['fg_code'][cut],
                                  'cut_width': item['cut_width'][cut],
                                  'cut_weight': item['cut_weight'][cut],
                                  'average_fc': item['average_fc'][cut],
                                  'remarks': item['remarks'][cut],
                                  'customer_short_name': item['customer_short_name'][cut]}
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def clean_filename(filename):
    # Define a regular expression pattern for allowed characters (letters, numbers, dots, underscores, and dashes)
    # Replace any character not in this set with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

# MAIN FLOW
# def merge_pattern()
       
def multistocks_cut(logger, finish, stocks, MATERIALPROPS, margin_df, prob_type):
    """
    to cut all possible stock with finish demand upto upperbound
    """
    bound = starting_bound
    logger.info(f"Starting bound: {starting_bound}")
    # SET UP
    steel = CuttingStocks(finish, stocks, MATERIALPROPS)
    steel.update(bound = starting_bound, margin_df = margin_df)
    steel.filter_stocks_by_group_standard()
    steel.check_division_stocks()
    
    st = {k for k in steel.filtered_stocks.keys()}
    logger.info(f"> After dividing stocks: {st}")
    
    cond = steel.set_prob(prob_type) #  "Dual" / "Rewind"
    final_solution_patterns = []
    
    while cond == True: # have stocks and need cut
        print("CUTTING ... ")
        len_last_sol = copy.deepcopy(len(final_solution_patterns))
        stt, final_solution_patterns, over_cut = steel.solve_prob("CBC") # neu ko reset, thi final_solution_patterns ket qua se duoc accumulate
        
        # Check solution_patterns - True if patterns empty and new pattern list == old pattern list
        check_solution_patterns = (not final_solution_patterns or len(final_solution_patterns) == len_last_sol)
        # IF - to increase bound
        if check_solution_patterns and bound == max_bound: # empty solution
            logger.info(f"Empty solution/or limited optimals, max bound")
            cond = False
            break #loop while
        elif check_solution_patterns and bound < max_bound: # empty solution but can increase bound
            bound += bound_step
            try: #  only able to refresh if len = last solution
                steel.refresh_stocks()
            except AttributeError: # empty solution in previous run
                pass
            
            finish_k = {k: v['need_cut'] for k, v in steel.prob.dual_finish.items() if v['need_cut'] > 0}
            logger.info(f" No optimal solution for needcut {finish_k}, increase to {bound} bound")
            steel.update_upperbound(bound)
            cond = True #continue to try new bound
        else: # have solution
            over_cut_rate = {k: round(over_cut[k]/(finish[k]['average FC']+1), 4) for k in over_cut.keys()}
            # update REMAINED STOCKS/FINISH and CONTINUE
            steel.refresh_stocks()
            steel.refresh_finish()
            
            logger.warning(f"!!! Status {stt}")
            logger.info(f">>> Take stock {[p['stock'] for p in final_solution_patterns]}")
            logger.info(f">>> Overcut amount {over_cut}")
            logger.info(f">>> Overcut ratio: {over_cut_rate}")
            
            # CONDITIONS to continue to cut
            # negative_over_cut_ratio = sum(value < stop_stock_ratio for value in over_cut_rate.values()) > 0
            # negative_over_cut_wg    = sum(value < stop_needcut_wg  for value in over_cut.values()     ) > 0
            # cond_fg = negative_over_cut_wg & 
            empty_fg = (not steel.prob.dual_finish) #empty
            if empty_fg:
                logger.info(f"!!! FINISH CUTTING")
            else:
                st_list = {k: v['need_cut'] for k, v in steel.prob.dual_finish.items()}
                logger.info(f">>> FG continue to cut {st_list}")
            
            re_stocks = [k for k in steel.prob.dual_stocks.keys()]
            empty_stocks = (not re_stocks) # true if empty stock list
            if empty_stocks: logger.info(f"!!! Out of stocks")
            else: logger.info(f">>> Remained Stocks {re_stocks}")
            
            cond = (not empty_stocks) & (not empty_fg)
    
    # FINALIZE results and RAISE Error for case absolute no solution
    if not final_solution_patterns and bound == max_bound:
        taken_stocks = []
        logger.warning("!!! No Solution")
        raise TypeError("Final_solution_patterns is empty, and reach MAX bound")
    else:
        taken_stocks    = [p['stock'] for p in final_solution_patterns]
        trimloss        = [p['trim_loss_pct'] for p in final_solution_patterns]
        stock_weight    = [p['stock_weight'] for p in final_solution_patterns]
        # logger
        logger.info(f">>> Total {len(final_solution_patterns)} Stocks are used, weighting {sum(stock_weight)}, average trim loss {round(np.mean(trimloss),3) if len(trimloss) > 0 else np.nan}")
        logger.info(f">>> with trim loss each MC {trimloss}")

    try:
        over_cut_rate = {k: round(over_cut[k]/(finish[k]['average FC']+1), 4) for k in over_cut.keys()}
        logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
    except UnboundLocalError: # NO solution
        over_cut = []
    
    # RETURN RESULTS
    if prob_type == "Rewind":
        remained_stocks = {
                    s: {**s_info}
                    for s, s_info in steel.prob.start_stocks.items()
                    if s not in taken_stocks
                }
        return final_solution_patterns, over_cut, taken_stocks, remained_stocks
    else:
        return final_solution_patterns, over_cut, taken_stocks
    
########################## START ##############################
today = datetime.datetime.today()
formatted_date = today.strftime("%y-%m-%d")

# START LOGGER
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'scr/log/batch3-small-med-big-{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# LOAD MASTER DATA
margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')

# LOAD JOB-LIST
logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
logger.info('*** LOAD JOB LIST ***')
with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
    job_list = json.load(file)

with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'r') as stocks_file:
    stocks_list =json.load(stocks_file)

with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
    finish_list = json.load(finish_file)

n_jobs = job_list['number of job']

### --- PARAMETER SETTING ---
i = 22
mc_ratio = 2.5
starting_bound = 2 #THAY DOI BOI NEEDCUT

# starting_bound = 2 work for i 22
added_stock_ratio = 0.1

# LOAD JOB INFO -- 1/2 LAYER
tasks = job_list['jobs'][i]['tasks']                        
materialprop_set = job_list['jobs'][i]['materialprop']
available_stock_qty = job_list['jobs'][i]['available_stock_qty']
MATERIALPROPS = stocks_list['materialprop_stock'][materialprop_set]['materialprop']
batch = MATERIALPROPS['code']

# LOAD STOCKS
og_stocks = stocks_list['materialprop_stock'][materialprop_set]['stocks'] # Original stocks available 
stocks_to_use = copy.deepcopy(og_stocks)
total_taken_stocks = []
taken_stocks =[]

# START JOB
logger.info("------------------------------------------------")
logger.info(f'*** START processing JOB {i} MATERIALPROPS:{batch} ***')
## Loop FINISH - each TASK (by CUSTOMER) in JOB - P1.0
for finish_item in finish_list['materialprop_finish'][materialprop_set]['group']:
    start_time = time.time()
    try:  ## Add try-catch for any error happen -> go to next job
        # SETUP
        over_cut = {}
        final_solution_patterns = []
        
        ## GET GROUP-NAME -- 1/2 LAYER - P1.1
        group_name = list(finish_item.keys())[0] #small-med-big
        logger.info(f" CUSTOMER_GROUP {group_name}")
        
        # Get BEGINNING FINISH list
        og_finish = finish_item[group_name]  # original finish with stock ratio < 0.5 (need cut can be positive)
        
        ### ADD FG CODE WITH NEED CUT > 0
        if added_stock_ratio <= stop_stock_ratio:
            filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut']/(v['average FC']+1) < stop_stock_ratio}
            if len(filtered_finish) < 3:
                finish = copy.deepcopy(og_finish)
            else:
                finish = copy.deepcopy(filtered_finish)
        else: finish = partite_finish(og_finish, added_stock_ratio)
            
        medium_mc_weight = np.percentile(sorted([v['Min_MC_weight'] for _, v in finish.items()]),50)
        
        partial_f_list = {k for k in finish.keys()}

        if not finish: #### empty list-> go to next gr
            pass 
        else: 
            total_need_cut_by_cust_gr = -sum(item["need_cut"] for item in finish.values() if item["need_cut"] < 0)
            logger.info(f"> Finished Goods key {partial_f_list}")
            logger.info(f"> Total Need Cut: {total_need_cut_by_cust_gr}")
            
            # OPERATOR OF COIL CENTER PRIORITY to cut until all FG DONE
            warehouse_order = create_warehouse_order(finish)
            logger.info(f"> WareHouse priority {warehouse_order}")
    
            # INITIATE stock from WAREHOUSE priority  P.0
            j = 0
            while j < no_warehouse: 
                filtered_stocks_by_wh = filter_stocks_by_wh(stocks_to_use,[warehouse_order[j]])
                # SELECT STOCKS by need cut
                partial_stocks = partite_stock(filtered_stocks_by_wh, total_need_cut_by_cust_gr * mc_ratio, group_name, medium_mc_weight)
                if len(partial_stocks.keys()) > 0:
                    st = {k for k in partial_stocks.keys()}
                    logger.info(f"> Number of stocks in {warehouse_order[j]} : {len(st)}")
                    coil_center_priority_cond = True
                    break
                else:
                    logger.warning(f'>>>> Out of stocks for for WH {warehouse_order[j]}')
                    j +=1
                    if j == no_warehouse: 
                        coil_center_priority_cond = False
            
            # ???NEXT WAREHOUSE
            next_warehouse_stocks = []
            while coil_center_priority_cond:
                if not next_warehouse_stocks: # Empty P3.1
                    pass 
                else:                # Refresh stocks P3.2
                    logger.info(f">> Cut in WH {warehouse_order[j]}")
                    logger.info(f">> for: {len(finish.keys())} FINISH  w {len(partial_stocks.keys())} MC")
                print(f"RUNNING.... in {warehouse_order[j]}")
                args_dict = {
                            'logger': logger,
                            'finish': finish,
                            'stocks': partial_stocks,
                            'MATERIALPROPS': MATERIALPROPS,
                            'margin_df': margin_df,
                            }
                
                # P4.1
                try: # START OPTIMIZATION
                    logger.info("*** NORMAL DUAL Case ***")
                    final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict,prob_type ="Dual")
                    ### Exclude taken_stocks out of stock_to_use only for dividing MC
                    stocks_to_use = refresh_stocks(taken_stocks, stocks_to_use)
                    
                except TypeError:  # raise in multistock_cut => ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                    if len(partial_stocks) == 1 and len(finish) == 1: #### SEMI CASE
                        logger.info('*** SEMI case *** 1 FG vs 1 Stock')
                        steel = SemiProb(partial_stocks, finish, MATERIALPROPS)
                        steel.update(margin_df)
                        steel.cut_n_create_new_stock_set()
                        #### Update lai stock
                        stocks_to_use.pop(list(partial_stocks.keys())[0]) # truong hop ko cat duoc 
                        stocks_to_use.update(steel.remained_stocks)     # thi 2 dong nay bu tru nhau
                        stocks_to_use.update(steel.taken_stocks)
                        try:
                            taken_stock = list(steel.taken_stocks.keys())[0]
                            weight_dict = {f: round(steel.cuts_dict[f] * finish[f]['width'] * partial_stocks[taken_stock]['weight']/partial_stocks[taken_stock]['width'],3) for f in steel.cuts_dict.keys()}
                            final_solution_patterns = [{"inventory_id": taken_stock, 
                                                        "stock_width":  steel.taken_stocks[taken_stock]['width'],
                                                        "stock_weight": steel.taken_stocks[taken_stock]['weight'],
                                                        "fg_code":{f: finish[f]['fg_codes'] for f in steel.cut_dict.keys()},
                                                        "customer_short_name": {f: finish[f]['customer_name'] for f in steel.cut_dict.keys()},
                                                        'cuts': steel.cut_dict,
                                                        "cut_weight": weight_dict,
                                                        "cut_width": {f:finish[f]['width'] for f in steel.cut_dict.keys()},
                                                        "explanation": "keep Semi",
                                                        "remark":"",
                                                        "cutting_date":"",
                                                        "trim_loss": 9999, "trim_loss_pct":9999,
                                                        # 'details': [{'order_no': f, 'width': finish[f]['width'], 'lines': steel.cut_dict[f]} for f in steel.cut_dict.keys()] 
                                                    }] #SEMI CASE - TRIM LOSS >>
                        except IndexError:
                            final_solution_patterns = []  
                    elif len(partial_stocks) == 1:                    #### REWIND CASE
                        logger.info("*** REWIND Case ***")
                        try:
                            final_solution_patterns, over_cut, taken_stocks, remained_stocks = multistocks_cut(**args_dict, prob_type="Rewind")
                            logger.info(f"REMAINED stocks: {remained_stocks}")
                            stocks_to_use.pop(list(partial_stocks.keys())[0]) # truong hop ko cat duoc 
                            stocks_to_use.update(remained_stocks)     # thi 2 dong nay bu tru nhau
                        except TypeError: pass 
                    else: pass
                        
                # COMPLETE CUTTING in current warehouse
                if not final_solution_patterns:
                    logger.warning(f"!!! NO solution/NO cutting at {warehouse_order[j]}")
                else: 
                    total_taken_stocks.append(taken_stocks)
                    # --- SAVE DF to EXCEL ---
                    end_time = time.time()
                    cleaned_materialprop_set = clean_filename(materialprop_set)
                    filename = f"scr/results/result-{cleaned_materialprop_set}-{group_name}-{warehouse_order[j]}.xlsx"
                    df = transform_to_df(final_solution_patterns)
                    df['time'] = end_time - start_time
                    df.to_excel(filename, index=False)
                
                    logger.info(f">>> SOLUTION {materialprop_set} for {group_name} at {warehouse_order[j]} saved  EXCEL file")
                    
                # GO TO next COIL CENTER priority if still ABLE
                if j < no_warehouse - 1: # go from ZERO 
                    j+=1
                    next_warehouse_stocks = filter_stocks_by_wh(stocks_to_use, [warehouse_order[j]])          #try to find STOCKS in next WH
                    try:
                        over_cut_rate = {f: round(over_cut[f]/(og_finish[f]['average FC']+1), 4) for f in over_cut.keys()}
                        # condition by stock ratio
                        # negative_over_cut_ratio = sum(value < stop_stock_ratio for value in over_cut_rate.values()) > 0
                        # negative_over_cut_kg = sum(value < stop_needcut_wg for value in over_cut.values()) > 0
                        
                        finish =  refresh_finish(finish, over_cut)
                        # condition fg continue to cut
                        has_negative_over_cut = (not finish) # empty
                        coil_center_priority_cond = (not has_negative_over_cut and (len(next_warehouse_stocks)!=0)) #can cat tiep va co coil o next priority
                        logger.info(f"??? Go to next warehouse: {coil_center_priority_cond}")
                        if coil_center_priority_cond:
                            # logger.info(f"overcut rate({over_cut_rate})")
                            finish =  refresh_finish(finish, over_cut)                                   # Remained need_cut finish to cut in next WH
                            f_list = {f : (finish[f]['need_cut']) for f in finish.keys()}               # need cut am
                            logger.info(f">>> Remained FG to cut in next WH: {f_list}")
                        # BACK TO P3.2
                    except TypeError or AttributeError:# overcut empty -> chua optimized duoc o coil center n-1
                        coil_center_priority_cond = (len(next_warehouse_stocks)!=0)            
                else: 
                    coil_center_priority_cond = False  # da het coil center de tim
                
                if coil_center_priority_cond:
                    total_over_cut = -sum(value < 0 for value in over_cut.values()) > 0
                    partial_stocks = partite_stock(next_warehouse_stocks, total_over_cut * mc_ratio, group_name, medium_mc_weight)
                    
            logger.info(f">>> END CUTTING STOCK-AFTER-CUT for {group_name}: {over_cut}")
            logger.info(f'--- DONE TASK for {group_name} ---')
    
    except Exception as e:
        logger.warning(f"Error with Customer {group_name}: {type(e)} {e}")
        logger.info(f"Occured on line {traceback.extract_tb(e.__traceback__)[-1]}")
        continue
        
logger.info('**** TEST JOB ENDED **** ')