# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import numpy as np
import os
import logging
import datetime
import json
import copy
import pulp
import math
import traceback
from model import CuttingStocks
from model import SemiProb
import re

# HELPERS
def clean_filename(filename):
    # Define a regular expression pattern for allowed characters (letters, numbers, dots, underscores, and dashes)
    # Replace any character not in this set with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

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

def filter_stocks_by_wh(data, warehouse_order):
    # Filter and order the dictionary
    return {k: v for w in warehouse_order for k, v in data.items() if v['warehouse'] == w}

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
            except KeyError:
                pass
            
    # Take only finish with negative need_cut
    re_finish = {k: v for k, v in finish.items() if v['need_cut']/(v['average FC']+1) < -0.01}
    if len(re_finish) >= 3:
        return re_finish
    else:
        re_finish = {k: v for k, v in finish.items() if v['need_cut']/(v['average FC']+1) < 0.3}
        return re_finish

def save_to_json(filename, data):
    with open(filename, 'w') as solution_file:
        json.dump(data, solution_file, indent=2)

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','cuts', "cut_w"]}
        for cut, line in item['cuts'].items():
            if line > 0:
                flattened_item = {**common_data, 'cuts': cut, 'lines': line, 'cut_weight': item['cut_w'][cut]}
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def refresh_stocks(taken_stocks,stocks_to_use):
    if not taken_stocks:
        remained_stocks = stocks_to_use
    else:
        # if any taken stock have Div in the name pop -Div, from stock to use ???
        taken_og = []
        for s in taken_stocks:
            if str(s).__contains__("-Di"):
                og_s = str(s).split("-Di")[0]
                taken_og.append(og_s)
        taken_stocks = taken_stocks + taken_og          
        remained_stocks = {
                s: {**s_info}
                for s, s_info in stocks_to_use.items()
                if s not in taken_stocks
            }
    return remained_stocks

def average_3m_fc(finish, over_cut):
    average_3fc = {
                key: (
                    sum(v for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3']) if not math.isnan(v)) /
                    sum(1 for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3']) if not math.isnan(v))
                ) if any(not math.isnan(v) for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3'])) else float('nan')
                for key in over_cut.keys()
            }
    return average_3fc
        
def multistocks_cut(logger, finish, stocks, PARAMS, margin_df, solver, prob_type):
    # pipeline to cut all possible stock with finish demand upto upperbound
    bound = 1
    steel = CuttingStocks(finish, stocks, PARAMS)
    steel.update(bound = bound, margin_df=margin_df)
    steel.check_division(s_customer_group=['small', 'medium', 'medium-big'], max_weight=8000)
    steel.filter_stocks() # flow khong consider min va max_weight
    cond = steel.set_prob(prob_type) #  "Dual" / "Rewind"
    final_solution_patterns = []
    while cond == True: # have stocks and need cut
        len_last_sol = len(final_solution_patterns)
        stt, final_solution_patterns, over_cut = steel.solve_prob(solver) # neu ko erase ket qua ben trong, thi patterns ket qua se duoc accumulate
    
        ins_bound = (not final_solution_patterns or len(final_solution_patterns) == len_last_sol)
        # IF - Bound
        if ins_bound and bound == max_bound: # empty solution
            logger.warning(f"Status Empty Solution, max bound")
            cond = False
            break #loop while
        elif ins_bound and bound < max_bound: # empty solution and can increase bound
            bound += 0.5
            try: #  only able to refresh if len = last solution
                steel.refresh_stocks()
            except AttributeError: # empty solution in previous run
                pass
            # Update bound of FG
            steel.F.update_bound(bound)
            logger.info(f">>>> No solution bound {bound - 0.5}, try increasing bound {bound}")
            cond = True #continue to try new bound
        else: # have solution
            logger.warning(f"Status {stt}")
            logger.info(f">>>> Take stock {[p['stock'] for p in final_solution_patterns]}")
            logger.info(f">>>> Overcut amount {over_cut}")
            steel.refresh_finish(over_cut)
            cond = steel.check_status()
        
    #IF finalize results
    if not cond:
        logger.info(f">>>> STOCKS USED {len(final_solution_patterns)}")
        trimloss=[p['trim_loss_pct'] for p in final_solution_patterns]
        logger.info(f">>>> TRIM LOSS PERCENT OF EACH STOCK {trimloss}, AVERAGE {np.mean(trimloss)}")
        mean_3fc = average_3m_fc(finish, over_cut)
        try:
            over_cut_rate = {k: round(over_cut[k]/mean_3fc[k], 4) for k in over_cut.keys()}
        except ZeroDivisionError:
            logger.warning(">>>> ZERO Forecast Data")
            over_cut_rate = {k: round(over_cut[k]/(mean_3fc[k]+1), 4) for k in over_cut.keys()}
        
        logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
    else: # update REMAINED STOCKS and CONTINUE
        steel.refresh_stocks()
        logger.info(f">>>> Stocks to continue to cut {[k for k in steel.prob.dual_stocks.keys()]}")
    
    # IF raise Error and return value
    if not final_solution_patterns and bound == max_bound:
        taken_stocks = []
        raise TypeError("Final_solution_patterns is empty, and reach MAX bound")
    elif (len(final_solution_patterns) == len_last_sol) and bound == max_bound:
        raise TypeError("Havent finished cutting, but no Optimal Solution and reach MAX bound")
    else:
        taken_stocks = [p['stock'] for p in final_solution_patterns]
    
    if prob_type == "Rewind":
        remained_stocks = {
                    s: {**s_info}
                    for s, s_info in steel.prob.start_stocks.items()
                    if s not in taken_stocks
                }
        return final_solution_patterns, over_cut, taken_stocks, remained_stocks
    else: 
        return final_solution_patterns, over_cut, taken_stocks

today = datetime.datetime.today()
formatted_date = today.strftime("%d-%m-%y")

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'scr/log/comb-cust-{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LOAD CONFIG & DATA
global max_bound
global no_warehouse

max_bound = float(os.getenv('MAX_BOUND', '5.0'))
no_warehouse = float(os.getenv('NO_WAREHOUSE', '3.0'))
print(f"MAX BOUND {max_bound}")
print(f"NO WH {no_warehouse}")

margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')

# LOAD JOB-LIST
logger.info('*** LOAD JOB LIST ***')
with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
    job_list = json.load(file)

with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'r') as stocks_file:
    stocks_list =json.load(stocks_file)

with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
    finish_list = json.load(finish_file)

n_jobs = job_list['number of job']

logger.info('--- PROCEED JOBS ---')
total_solution_json = {"date": formatted_date,
                        "number_params":n_jobs,
                        "status":200,
                        'solution':[]
                        }

# RUN JOB-LIST
i = 5

# LOAD JOB INFO -- 1/2 LAYER                           
param_set = job_list['jobs'][i]['param']
stocks_available = job_list['jobs'][i]['stocks_available']
tasks = job_list['jobs'][i]['tasks']
PARAMS = stocks_list['param_finish'][param_set]['param']
batch = PARAMS['code']

# LOAD STOCKS
og_stocks = stocks_list['param_finish'][param_set]['stocks'] # Original stocks available - phan biet vs stocks se dung cho tung customer
stocks_to_use = copy.deepcopy(og_stocks)
    
# START
logger.info("------------------------------------------------")
logger.info(f'-> START processing JOB {i} PARAMS: [{batch}]')
total_taken_stocks = []
taken_stocks =[]

## Loop FINISH - each TASK (by CUSTOMER) in JOB - P1.0
for finish_item in finish_list['param_finish'][param_set]['customer']: 
    try:  ## Add try-catch for any error happen -> go to next job
        ## GET CUSTOMER-NAME -- 1/2 LAYER - P1.1
        customer_name = list(finish_item.keys())[0]
        finish_by_gr = finish_item[customer_name]  ## original finish with stock ratio < 0.5 ( need cut can be positive)
        logger.info(f"->> for CUSTOMER {customer_name}")
        
        # Get beginning finish list
        og_finish = finish_item[customer_name]  # original finish with stock ratio < 0.5 (need cut can be positive)
        over_cut = {}
        final_solution_patterns = []
        
        ### if NUMBER NEED CUT FG < 0 small, take all
        filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut']/(v['average FC']+1) < 0.0}
        if len(filtered_finish) < 3:
            finish = copy.deepcopy(og_finish)
        else:
            finish = copy.deepcopy(filtered_finish)
        if not finish:
            pass #### go to next min-max gr
        else:
            warehouse_order = create_warehouse_order(finish)
        logger.info(f"->>> WareHouse priority {warehouse_order}")
        logger.info(f"->>> Number of stocks left: {len(stocks_to_use)}")
        
        ### SORT STOCK BY coil center priority, changed by customer preference
        j = 0 # j = {0, 1, 2}
        while j < no_warehouse: ### INITIATE stock from WH priority  P2.0
            stocks_by_wh = filter_stocks_by_wh(stocks_to_use,[warehouse_order[j]])
            if len(stocks_by_wh.keys()) > 0:
                logger.info(f"->>> cut in WH {warehouse_order[j]}")
                coil_center_priority_cond = True
                break
            else:
                logger.warning(f'>>>> Out of stocks for for WH {warehouse_order[j]}')
                j +=1
                if j == no_warehouse: 
                    coil_center_priority_cond = False
        logger.info(f"->>> SUB-TASK: {len(finish.keys())} FINISH  w {len(stocks_by_wh.keys())} MC")
        nx_wh_stocks = []      
        while coil_center_priority_cond: ### OPERATOR COIL CENTER to cut until all FG DONE P3.0
            if not nx_wh_stocks: # Empty P3.1
                pass 
            else:                # Refresh stocks P3.2
                stocks_by_wh = copy.deepcopy(nx_wh_stocks)
                logger.info(f"->>> cut in WH {warehouse_order[j]}")
                logger.info(f"Cut for: {len(finish.keys())} FINISH  w {len(stocks_by_wh.keys())} MC")
            # bound = 1
            print(f"RUNNING.... in {warehouse_order[j]}")
            # while bound <= max_bound:    # OPERATOR UPPER BOUND = {0.5, 1, 1.5} P4.0
            args_dict = {
                        'logger': logger,
                        'finish': finish,
                        'stocks': stocks_by_wh,
                        'PARAMS': PARAMS,
                        'margin_df': margin_df,
                        }
            
            logger.info("*** NORMAL DUAL case - GLPK ***")
            try:              # P4.1
                final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict, solver = None, prob_type="Dual")
                ### Exclude taken_stocks out of stock_to_use only for dual case
                stocks_to_use = refresh_stocks(taken_stocks, stocks_to_use)
            except TypeError or pulp.apis.core.PulpSolverError:  # raise in multistock_cut => ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                if len(stocks_by_wh) == 1 and len(finish) == 1: #### SEMI CASE
                    logger.info('*** SEMI case *** 1 FG vs 1 Stock')
                    steel = SemiProb(stocks_by_wh, finish, PARAMS)
                    steel.update(margin_df)
                    steel.cut_n_create_new_stock_set()
                    #### Update lai stock
                    stocks_to_use.pop(list(stocks_by_wh.keys())[0]) # truong hop ko cat duoc 
                    stocks_to_use.update(steel.remained_stocks)     # thi 2 dong nay bu tru nhau
                    stocks_to_use.update(steel.taken_stocks)
                    try:
                        taken_stock = list(steel.taken_stocks.keys())[0]
                        final_solution_patterns = [{"inventory_id": taken_stock, 
                                                    "stock_width":  steel.taken_stocks[taken_stock]['width'],
                                                    "stock_weight": steel.taken_stocks[taken_stock]['weight'],
                                                    "customer_short_name": customer_name,
                                                    "explanation": "Semi Cut",
                                                    "remark":"",
                                                    "cutting_date":"",
                                                    "trim_loss_mm": 9999, "trim_loss_pct": 9999,
                                                    'cuts': steel.cut_dict,
                                                    'details': [{'order_no': f, 'width': finish[f]['width'], 'lines': steel.cut_dict[f]} for f in steel.cut_dict.keys()] 
                                                 }] #SEMI CASE - TRIM LOSS >>
                    except IndexError:
                        final_solution_patterns = []
                        
                elif len(stocks_by_wh) == 1: #### REWIND 
                    logger.info("*** REWIND case ***")
                    try:
                        final_solution_patterns, over_cut, taken_stocks, remained_stocks = multistocks_cut(**args_dict,solver =None, prob_type="Rewind")
                        logger.info(f"REMAINED stocks: {remained_stocks}")
                        stocks_to_use.pop(list(stocks_by_wh.keys())[0]) # truong hop ko cat duoc 
                        stocks_to_use.update(remained_stocks)     # thi 2 dong nay bu tru nhau
                    except TypeError or pulp.apis.core.PulpSolverError: ### empty final_sol_pattern
                        logger.warning("*** No Solution for Rewind case - GLPK ***")
                        logger.info("*** NORMAL DUAL case - CBC ***")
                        try:
                            final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict, solver = "CBC", prob_type="Dual")
                            stocks_to_use = refresh_stocks(taken_stocks, stocks_to_use)
                        except TypeError: #empty solution
                            pass
                
                else: # DUAL CBC
                    logger.info("*** NORMAL DUAL case - try CBC ***")
                    ### NO SOLUTION for REWIND va SEMI -> reset bound, try sub-optimal CBC - go to P4.1
                    try:
                        final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict, solver = "CBC", prob_type="Dual")
                        stocks_to_use = refresh_stocks(taken_stocks, stocks_to_use)
                    except TypeError:
                        pass
            
            # Complete Cutting in current warehouse
            if not final_solution_patterns:
                logger.warning(f"->>>> The solution at WH {warehouse_order[j]} is empty.")
            else: 
                ### neu co nghiem thi break while bound < 4 
                total_taken_stocks.append(taken_stocks)
                
                # --- SAVE to DF ---
                df = transform_to_df(final_solution_patterns)
                cleaned_param_set = clean_filename(param_set)
                filename = f"scr/results/comb-cust-{cleaned_param_set}-{customer_name}-{warehouse_order[j]}.xlsx"
                df.to_excel(filename, index=False)
            
                logger.info(f"->>>> The solution {param_set} for {customer_name} at {warehouse_order[j]} saved  EXCEL file")
                
            # To nex COIL CENTER priority if still ABLE
            if j < no_warehouse - 1: # go from ZERO 
                j+=1
                nx_wh_stocks = filter_stocks_by_wh(stocks_to_use, [warehouse_order[j]])          #try to find STOCKS in next WH
                try:
                    has_negative_over_cut = any(value < 0 for value in over_cut.values())
                    coil_center_priority_cond = (has_negative_over_cut & (len(nx_wh_stocks)!=0)) #can cat tiep va co coil o next priority
                    logger.info(f"->>> ? Go to next warehouse: {coil_center_priority_cond}")
                    if coil_center_priority_cond:
                        finish =  refresh_finish(finish, over_cut)                                   # Remained need_cut finish to cut in next WH
                        f_list = {f : (finish[f]['need_cut']) for f in finish.keys()}
                        logger.info(f"->>> Remained finish to cut in next WH: {f_list}")
                    # BACK TO P3.2
                except TypeError or AttributeError:                                              # overcut empty -> chua optimized duoc o coil center n-1
                    coil_center_priority_cond = (len(nx_wh_stocks)!=0)            
            else: 
                coil_center_priority_cond = False  # da het coil center de tim
                
        logger.info(f">>> END CUTTING STOCK-AFTER-CUT for {customer_name}: {over_cut}")
        logger.info(f'--- DONE TASK for {customer_name} ---')
    except Exception as e:
        logger.warning(f"Error with Customer {customer_name}: {type(e)} {e}")
        logger.info(f"Occured on line {traceback.extract_tb(e.__traceback__)[-1]}")
        continue
        
logger.info('**** TEST JOB ENDED **** ')