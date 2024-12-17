# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import numpy as np
import traceback
import logging
import datetime
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

max_coil_weight = float(os.getenv('MAX_WEIGHT_MC_DIV', '7000'))
max_bound    = float(os.getenv('MAX_BOUND', '3.0'))
no_warehouse = float(os.getenv('NO_WAREHOUSE', '3.0'))
added_stock_ratio_avg_fc = float(os.getenv('ADDED_ST_RATIO_AVG_FC', '2000'))
 
# Group to div stock >8000
customer_gr = os.getenv('CUSTOMER_GR')
if customer_gr:
    customer_group = customer_gr.split(',')
else:
    customer_group = ['small','small-medium','medium']

# # User adjustably  
# mc_ratio        = float(os.getenv('MC_RATIO', '2.5'))
# added_stock_ratio = float(os.getenv('ST_RATIO', '-0.02'))

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

def partial_stock(stocks,allowed_cut_weight):
    """Take amount of stock equals 2.5 to 3.0 need cut
    Args:
        stocks (dict): _description_
        allowed_cut_weight (float): _description_

    Returns:
        partial_stocks: dict
    """
    stocks = dict(sorted(stocks.items(), key=lambda x: (x[1]['weight'], x[1]['receiving_date']), reverse= True))
    partial_stocks = {}
    accumulated_weight = 0
    
    for s, sinfo in stocks.items():
        accumulated_weight += sinfo['weight']
        if allowed_cut_weight * 1.1 <= accumulated_weight:
            if len(partial_stocks) ==0:
                partial_stocks[s] = {**sinfo}
            else:
                break
        else:
            partial_stocks[s] = {**sinfo}
             
    return partial_stocks

def partial_finish(finish, stock_ratio):
    """_Select more FG codes (finish) below the indicated stock ratio to reduce trim loss _

    Args:
        finish (_type_): _description_
        stock_ratio (_type_): _description_

    Returns:
        partial_finish: the proportion of finish has the stock ratio as required
    """
    partial_pos_finish = {}
    added_stock_ratio_avg_fc = [3000, 1500, 200]
    for avg_fc in added_stock_ratio_avg_fc:
        print(f"range forecast {avg_fc}")
        for f, finfo in finish.items():
            average_fc = max(finfo.get('average FC', 0), 1)
            fg_ratio = finfo['need_cut'] / average_fc
            if (0 <= fg_ratio <= stock_ratio and round(finfo['average FC']) >= avg_fc):
                partial_pos_finish[f] = finfo.copy()
                print(f"LENG {len(partial_pos_finish)}")
        
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
        def_avg_fc = def_avg_fc if 'def_avg_fc' in locals() else 200

        # Check conditions for partial finishes
        if (
            fg_ratio < 0 
            or (0 <= fg_ratio <= stock_ratio and round(average_fc) >= def_avg_fc)
        ):
            partial_finish[f] = finfo.copy()

                    
    return partial_finish

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
    re_finish = {k: v for k, v in finish.items() if v['need_cut']/(v['average FC']+1) < -0.02}
    if len(re_finish) >= 3:
        return re_finish
    else:
        re_finish = {k: v for k, v in finish.items() if v['need_cut']/(v['average FC']+1) < 0.3}
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

def average_3m_fc(finish):
    average_3fc = {f: f_info['average FC']+1 for f, f_info in finish.items()} # plus 1 for zero forecast
    return average_3fc

# Save File
def save_to_json(filename, data):
    with open(filename, 'w') as solution_file:
        json.dump(data, solution_file, indent=2)

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','cuts', "cut_w", "remark"]}
        for cut, line in item['cuts'].items():
            if line > 0:
                flattened_item = {**common_data, 'cuts': cut, 
                                  'lines': line,
                                  'cut_weight': item['cut_w'][cut],
                                  'remarks': item['remarks'][cut]}
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def clean_filename(filename):
    # Define a regular expression pattern for allowed characters (letters, numbers, dots, underscores, and dashes)
    # Replace any character not in this set with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

# Control flow 
def find_optimal_bound_step(test_finish, test_stocks, MATERIALPROPS, margin_df):
    # Find best step, which usually take more stocks in the first round
    if len(test_finish) > 5:
        bound_range = [0.5, 0.75]
    else:
        bound_range = [0.5, 0.75, 1.0, 1.5]
    
    len_first_round_sol = [] # take 1st solution, the longer length is better
    print("Find bound step (already div stocks)")
    
    for b in bound_range:
        print(f"try-bound {b}")
        test_steel = CuttingStocks(test_finish, test_stocks, MATERIALPROPS)
        test_steel.S.update_min_margin(margin_df = margin_df)
        test_steel.filter_stocks() # flow khong consider min va max_weight
        test_steel.check_division_stocks()
        test_steel.F.reverse_need_cut_sign()
        _ = test_steel.set_prob("Dual")
        test_steel.update_upperbound(b)
        _, final_solution_patterns, _ = test_steel.solve_prob("CBC")
        print(f"Value at bound {b}: {len(final_solution_patterns)}")
        len_first_round_sol.append(len(final_solution_patterns))
        # reset
        del test_steel
        
    if all(x == 0 for x in len_first_round_sol):
        optimal_bound_step = 1
        
    elif all(x == len_first_round_sol[0] for x in len_first_round_sol):
        # print("All elements are equal")
        optimal_bound_step = 0.75
    else:
        # Not all elements are equal, find the index of the max value
        print(len_first_round_sol)
        max_value = max(len_first_round_sol)

        # Find the last index of the maximum value
        # max_index = len(len_first_round_sol) - 1 - len_first_round_sol[::-1].index(max_value)
        max_index = len_first_round_sol.index(max_value)
        
        optimal_bound_step = bound_range[max_index]
        print(f"position {max_index}")
    
    return optimal_bound_step
       
def multistocks_cut(logger, finish, stocks, MATERIALPROPS, margin_df, prob_type):
    """
    to cut all possible stock with finish demand upto upperbound
    """
    # test_finish = copy.deepcopy(finish)
    # test_stocks = copy.deepcopy(stocks)
    
    # try: 
    #     logger.info("Find bound_step")
    #     optimal_bound_step = find_optimal_bound_step(test_finish, test_stocks, MATERIALPROPS, margin_df)
    # except Exception:
    #     optimal_bound_step = 1
        
    optimal_bound_step = 0.75
        
    bound = optimal_bound_step
    logger.info(f"Optimal bound step: {optimal_bound_step}")
    
    steel = CuttingStocks(finish, stocks, MATERIALPROPS)
    steel.update(bound = optimal_bound_step, margin_df = margin_df)
    
    steel.filter_stocks()
    steel.check_division_stocks()
    
    st = {k for k in steel.filtered_stocks.keys()}
    logger.info(f"->>> After dividing stocks: {st}")
    
    cond = steel.set_prob(prob_type) #  "Dual" / "Rewind"
    final_solution_patterns = []
    
    while cond == True: # have stocks and need cut
        len_last_sol = len(final_solution_patterns)
        stt, final_solution_patterns, over_cut = steel.solve_prob("CBC") # neu ko erase ket qua ben trong, thi patterns ket qua se duoc accumulate
        ins_bound = (not final_solution_patterns or len(final_solution_patterns) == len_last_sol)
        # IF - Bound
        if ins_bound and bound == max_bound: # empty solution
            logger.info(f"Empty solution/or limited optimals, max bound")
            cond = False
            break #loop while
        elif ins_bound and bound < max_bound: # empty solution and can increase bound
            bound += optimal_bound_step
            try: #  only able to refresh if len = last solution
                steel.refresh_stocks()
            except AttributeError: # empty solution in previous run
                pass
            finish_k = {k for k in steel.prob.dual_finish.keys()}
            logger.info(f" No solution/or limited optimals for {finish_k}, increase to {bound} bound")
            steel.update_upperbound(bound)
            cond = True #continue to try new bound
            
        else: # have solution
            logger.warning(f"Status {stt}")
            logger.info(f">>>> Take stock {[p['stock'] for p in final_solution_patterns]}")
            logger.info(f">>>> Overcut amount {over_cut}")
            steel.refresh_finish(over_cut)
            cont_cut = steel.check_status() # do we need to continue to cut
            mean_3fc = average_3m_fc(finish)
            try:
                over_cut_rate = {k: round(over_cut[k]/mean_3fc[k], 4) for k in over_cut.keys()}
            except ZeroDivisionError:
                over_cut_rate = {k: round(over_cut[k]/(mean_3fc[k]+1), 4) for k in over_cut.keys()}
                 
            negative_over_cut_ratio = sum(value < -0.02 for value in over_cut_rate.values()) > 0
            negative_over_cut_wg = sum(value <- 100 for value in over_cut.values()) > 0
            
            has_negative_over_cut = negative_over_cut_wg & negative_over_cut_ratio
           
            cond = has_negative_over_cut & cont_cut
            if cond:
                # update REMAINED STOCKS and CONTINUE
                steel.refresh_stocks()
                logger.info(f">>>> Stocks to continue to cut {[k for k in steel.prob.dual_stocks.keys()]}")
    
    #IF finalize results
    logger.info(f">>>> STOCKS USED {len(final_solution_patterns)}")
    trimloss=[p['trim_loss_pct'] for p in final_solution_patterns]
    st_wg = [p['stock_weight'] for p in final_solution_patterns]
    logger.info(f">>>> TOTAL USED STOCK WEIGHT {sum(st_wg)}")
    logger.info(f">>>> TRIM LOSS PERCENT OF EACH STOCK {trimloss}, AVERAGE {np.mean(trimloss) if len(trimloss) > 0 else np.nan}")
    mean_3fc = average_3m_fc(finish)
    try:
        over_cut_rate = {k: round(over_cut[k]/mean_3fc[k], 4) for k in over_cut.keys()}
    except ZeroDivisionError:
        logger.warning(">>>> ZERO Forecast Data")
        over_cut_rate = {k: round(over_cut[k]/(mean_3fc[k]+1), 4) for k in over_cut.keys()}
    
    logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
    
    # IF raise Error for case absolute no solution
    if not final_solution_patterns and bound == max_bound:
        taken_stocks = []
        logger.warning("No Solution")
        raise TypeError("Final_solution_patterns is empty, and reach MAX bound")
    else:
        taken_stocks = [p['stock'] for p in final_solution_patterns]
    
    # IF WHICH return
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

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'scr/log/batch-3-{formatted_date}.log', level=logging.INFO, 
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

logger.info('--- PROCEED JOBS w WEIGHT---')
total_solution_json = {"date": formatted_date,
                        "number_materialprops":n_jobs,
                        "status":200,
                        'solution':[]
                        }

# RUN JOB-LIST
i = 4
mc_ratio = 2.5
added_stock_ratio = -0.02
# added_stock_ratio = 0.3

# LOAD JOB INFO -- 1/2 LAYER                           
materialprop_set = job_list['jobs'][i]['materialprop']
stocks_available = job_list['jobs'][i]['stocks_available']
tasks = job_list['jobs'][i]['tasks']
MATERIALPROPS = stocks_list['materialprop_stock'][materialprop_set]['materialprop']
batch = MATERIALPROPS['code']

# LOAD STOCKS
og_stocks = stocks_list['materialprop_stock'][materialprop_set]['stocks'] # Original stocks available 
stocks_to_use = copy.deepcopy(og_stocks)

# START
logger.info("------------------------------------------------")
logger.info(f'-> START processing JOB {i} MATERIALPROPS: [{batch}] - CBC only if NO SOLS')
total_taken_stocks = []
taken_stocks =[]

## Loop FINISH - each TASK (by CUSTOMER) in JOB - P1.0
for finish_item in finish_list['materialprop_finish'][materialprop_set]['group']:
    try:  ## Add try-catch for any error happen -> go to next job
        # SETUP
        over_cut = {}
        final_solution_patterns = []
        
        ## GET GROUP-NAME -- 1/2 LAYER - P1.1
        group_name = list(finish_item.keys())[0] #small-med-big
        logger.info(f"->> for CUSTOMER_GR {group_name}")
        
        # Get BEGINNING FINISH list
        og_finish = finish_item[group_name]  # original finish with stock ratio < 0.5 (need cut can be positive)
        
        ### ?ADD FG CODE WITH NEED CUT > 0
        if added_stock_ratio == -0.02:
            filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut']/(v['average FC']+1) < -0.02}
            if len(filtered_finish) < 3:
                finish = copy.deepcopy(og_finish)
            else:
                finish = copy.deepcopy(filtered_finish)
        else:
            finish = partial_finish(og_finish, added_stock_ratio)
        
        partial_f_list = {k for k in finish.keys()}
        logger.info(f"Finished Goods key {partial_f_list}")
        
        if not finish:
            pass #### go to next gr
        else:
            warehouse_order = create_warehouse_order(finish)
        logger.info(f"->>> WareHouse priority {warehouse_order}")
        
        ### SORT STOCK BY 
        # coil center priority, changed by customer preference
        j = 0
        while j < no_warehouse: 
            ### INITIATE stock from WH priority  P2.0
            filtered_stocks_by_wh = filter_stocks_by_wh(stocks_to_use,[warehouse_order[j]])
            
            # partial stock by need cut
            total_need_cut_by_cust_gr = -sum(item["need_cut"] for item in finish.values() if item["need_cut"] < 0)
            logger.info(f"Total Need Cut: {total_need_cut_by_cust_gr}")
            partial_stocks = partial_stock(filtered_stocks_by_wh, total_need_cut_by_cust_gr * mc_ratio)
            
            if len(partial_stocks.keys()) > 0:
                st = {k for k in partial_stocks.keys()}
                logger.info(f"->>> Number of Total stocks in {warehouse_order[j]} : {len(st)}")
                coil_center_priority_cond = True
                break
            else:
                logger.warning(f'>>>> Out of stocks for for WH {warehouse_order[j]}')
                j +=1
                if j == no_warehouse: 
                    coil_center_priority_cond = False
        next_warehouse_stocks = []
        
        while coil_center_priority_cond: ### OPERATOR COIL CENTER to cut until all FG DONE P3.0
            if not next_warehouse_stocks: # Empty P3.1
                pass 
            else:                # Refresh stocks P3.2
                logger.info(f"->>> cut in WH {warehouse_order[j]}")
                logger.info(f"Cut for: {len(finish.keys())} FINISH  w {len(partial_stocks.keys())} MC")
            
            print(f"RUNNING.... in {warehouse_order[j]}")
            args_dict = {
                        'logger': logger,
                        'finish': finish,
                        'stocks': partial_stocks,
                        'MATERIALPROPS': MATERIALPROPS,
                        'margin_df': margin_df,
                        }
            
            logger.info("*** NORMAL DUAL Case ***")
            # P4.1
            try:
                final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict,
                                                                                  prob_type ="Dual"
                                                                                  )
                ### Exclude taken_stocks out of stock_to_use only for dividing MC
                stocks_to_use = refresh_stocks(taken_stocks, stocks_to_use)
                st = {k for k in stocks_to_use.keys()}
                logger.info(f"->>> Number of After-cut stocks: {st}")
               
            except TypeError:  # raise in multistock_cut => ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                if len(partial_stocks) == 1 and len(finish) == 1: 
                    #### SEMI CASE
                    logger.info('*** SEMI case *** 1 FG vs 1 Stock')
                    steel = SemiProb(stocks_by_wh, finish, MATERIALPROPS)
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
                                                    "customer_short_name": {f:finish[f]['customer_name'] for f in steel.cut_dict.keys()},
                                                    "explanation": "Semi Cut",
                                                    "remark":"",
                                                    "cutting_date":"",
                                                    "trim_loss": 9999, "trim_loss_pct": 9999,
                                                    'cuts': steel.cut_dict,
                                                    "cuts_width": {f:finish[f]['width'] for f in steel.cut_dict.keys()}
                                                    # 'details': [{'order_no': f, 'width': finish[f]['width'], 'lines': steel.cut_dict[f]} for f in steel.cut_dict.keys()] 
                                                 }] #SEMI CASE - TRIM LOSS >>
                    except IndexError:
                        final_solution_patterns = []
                        
                elif len(partial_stocks) == 1: #### REWIND 
                    logger.info("*** REWIND case ***")
                    try:
                        final_solution_patterns, over_cut, taken_stocks, remained_stocks = multistocks_cut(**args_dict,optimal_bound_step=0.5,prob_type="Rewind")
                        logger.info(f"REMAINED stocks: {remained_stocks}")
                        stocks_to_use.pop(list(partial_stocks.keys())[0]) # truong hop ko cat duoc 
                        stocks_to_use.update(remained_stocks)     # thi 2 dong nay bu tru nhau
                    except TypeError: pass 
                
                else: pass
                    
            # Complete Cutting in current warehouse
            if not final_solution_patterns:
                logger.warning(f"->>>> NO solution/NO cut at WH {warehouse_order[j]}")
            else: 
                ### neu co nghiem thi break while bound < 4 
                total_taken_stocks.append(taken_stocks)
                
                # --- SAVE to DF ---
                df = transform_to_df(final_solution_patterns)
                cleaned_materialprop_set = clean_filename(materialprop_set)
                filename = f"scr/results/result-{cleaned_materialprop_set}-{group_name}-{warehouse_order[j]}.xlsx"
                df.to_excel(filename, index=False)
            
                logger.info(f"->>>> SOLUTION {materialprop_set} for {group_name} at {warehouse_order[j]} saved  EXCEL file")
                
            # To nex COIL CENTER priority if still ABLE
            if j < no_warehouse - 1: # go from ZERO 
                j+=1
                next_warehouse_stocks = filter_stocks_by_wh(stocks_to_use, [warehouse_order[j]])          #try to find STOCKS in next WH
                
                try:
                    mean_og_3fc = average_3m_fc(og_finish)
                    try:
                        over_cut_rate = {k: round(over_cut[k]/mean_og_3fc[k], 4) for k in over_cut.keys()}
                    except ZeroDivisionError:
                        over_cut_rate = {k: round(1*over_cut[k]/(mean_og_3fc[k]+1), 4) for k in over_cut.keys()}

                    # condition by stock ratio
                    negative_over_cut_ratio = sum(value < -0.02 for value in over_cut_rate.values()) > 0
                    
                    # condition by kg
                    negative_over_cut_kg = sum(value <- 100 for value in over_cut.values()) > 0
                    
                    has_negative_over_cut = negative_over_cut_ratio & negative_over_cut_kg
                    coil_center_priority_cond = (has_negative_over_cut and (len(next_warehouse_stocks)!=0)) #can cat tiep va co coil o next priority
                    
                    logger.info(f"->>> ? Go to next warehouse: {coil_center_priority_cond}")
                    if coil_center_priority_cond:
                        # logger.info(f"overcut rate({over_cut_rate})")
                        finish =  refresh_finish(finish, over_cut)                                   # Remained need_cut finish to cut in next WH
                        f_list = {f : (finish[f]['need_cut']) for f in finish.keys()}               # need cut am
                        logger.info(f"->>> Remained finish to cut in next WH: {f_list}")
                    # BACK TO P3.2
                except TypeError or AttributeError:                                              # overcut empty -> chua optimized duoc o coil center n-1
                    coil_center_priority_cond = (len(next_warehouse_stocks)!=0)            
            else: 
                coil_center_priority_cond = False  # da het coil center de tim
            
            if coil_center_priority_cond:
                total_over_cut = -sum(value < 0 for value in over_cut.values()) > 0
                partial_stocks = partial_stock(next_warehouse_stocks, total_over_cut * mc_ratio)
                
        logger.info(f">>> END CUTTING STOCK-AFTER-CUT for {group_name}: {over_cut}")
        logger.info(f'--- DONE TASK for {group_name} ---')
    except Exception as e:
        logger.warning(f"Error with Customer {group_name}: {type(e)} {e}")
        logger.info(f"Occured on line {traceback.extract_tb(e.__traceback__)[-1]}")
        continue
        
logger.info('**** TEST JOB ENDED **** ')