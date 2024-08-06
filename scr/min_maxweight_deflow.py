# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import logging
import datetime
import json
import os
import copy
from model.O42_cutting_stocks import Cuttingtocks
from model.O41_semi_prob import SemiProb
import re

def clean_filename(filename):
    # Define a regular expression pattern for allowed characters (letters, numbers, dots, underscores, and dashes)
    # Replace any character not in this set with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

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

def filter_stocks_by_wh(data, warehouse_order):
    # Filter and order the dictionary
    return {k: v for w in warehouse_order for k, v in data.items() if v['warehouse'] == w}

def refresh_finish(finish, over_cut):
    # Update need cut
    for f in over_cut.keys(): # neu f khong co trong over_cut thi tuc la finish[f] chua duoc xu ly
        if over_cut[f] < 0:
            finish[f]['need_cut'] = -over_cut[f]
        else: 
            finish[f]['need_cut'] = 0
    
    return finish

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','cuts',"cut_w"]}
        for cut, line in item['cuts'].items():
            if line > 0:
                flattened_item = {**common_data, 'cuts': cut, 'lines': line}
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def multistocks_pipeline(logger, finish, stocks, min_w,max_w, PARAMS, bound, margin_df, prob_type):
    # pipeline to cut all possible stock with finish demand upto upperbound
    steel = Cuttingtocks(finish, stocks, PARAMS)
    steel.update(bound = bound, margin_df=margin_df)
    steel.filter_stocks_by_min_max_weight(min_w,max_w)
    cond =steel.set_prob(prob_type) # set condition to set prob_type = "Dual" / "Rewind"
    final_solution_patterns = []
    while cond == True: 
        # loop to cut as many FG as possible with < 4% trim loss
        stt, final_solution_patterns, over_cut = steel.solve_prob()
        if not final_solution_patterns: # empty solution
            logger.warning(f"Status Infeasible")
            break
        else:
            logger.warning(f"Status {stt}")
            logger.info(f'>>>> Take stock {[p['stock'] for p in final_solution_patterns]}')
            logger.info(f'>>>> Overcut amount {over_cut}')
            cond = steel.check_status()
            if not cond:
                over_cut_rate = {key: round(over_cut[key]/finish[key]['fc1'], 4) for key in over_cut.keys()}
                logger.info(f">>>> STOCKS USED {len(final_solution_patterns)}")
                logger.info(f'>>>> TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}')        
                logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
            else:
                steel.refresh_data()
                logger.info(f">>>> Stocks to continue to cut {steel.prob.dual_stocks.keys()}")
    # UPDATE REMAINED STOCKS IF CONTINUE
    if not final_solution_patterns:
        # Handle the case when final_solution_patterns is empty
        taken_stocks = []
        raise ValueError("final_solution_patterns is empty")
    else:
        taken_stocks = [p['stock'] for p in final_solution_patterns]
    
    return final_solution_patterns, over_cut, taken_stocks

today = datetime.datetime.today()
formatted_date = today.strftime("%d-%m-%y")

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'cutting_stocks_{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LOAD CONFIG & DATA
margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')

logger.info('*** LOAD JOB LIST ***')
# LOAD JOB-LIST
with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
    job_list = json.load(file)

with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'r') as stocks_file:
    stocks_list =json.load(stocks_file)

with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
    finish_list = json.load(finish_file)

n_jobs = job_list['number of job']

logger.info('--- PROCEED JOBS ---')
for i in range(n_jobs): # loop for each JOB - PARAM set
    # LOAD JOB INFO -- 1/3 LAYER
    param_set = job_list['jobs'][i]['param']
    stocks_available = job_list['jobs'][i]['stocks_available']
    tasks = job_list['jobs'][i]['tasks']
    PARAMS = stocks_list['param_finish'][param_set]['param']
    batch = PARAMS['code']

    # LOAD STOCKS
    og_stocks = stocks_list['param_finish'][param_set]['stocks'] # original stocks available - phan biet vs stocks se dung cho tung customer
    stocks_to_use = copy.deepcopy(og_stocks)
    
    # START
    logger.info(f'-> START processing JOB {i} PARAMS: [{batch}]')
    total_taken_stocks = []
    taken_stocks =[]
    for finish_item in finish_list['param_finish'][param_set]['customer']: ## loop FINISH - each TASK (by CUSTOMER) in JOB 
        ## GET CUSTOMER-NAME -- 2/3 LAYER
        customer_name = list(finish_item.keys())[0]
        finish_by_gr = finish_item[customer_name]  ## original finish with stock ratio < 0.5 ( need cut can be positive)
        
        logger.info(f"->> for CUSTOMER {customer_name}")
        
        for gr_dict in finish_by_gr: ### loop FINISH by min-max-weight-gr  -- 3/3 LAYER
            ### RESET
            over_cut = {}
            final_solution_patterns = []
            
            k = list(gr_dict.keys())[0] # min - max weight 
            logger.info(f"->>> in MIN-MAX GR {k}")
            values = str(k).split("-")

            ### Assign the values to min and max
            min_weight = float(values[0])
            max_weight = float(values[1])
            
            og_finish = gr_dict[k]
            
            ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
            filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
            if len(filtered_finish) < 3:
                finish = copy.deepcopy(og_finish)
            else:
                finish = copy.deepcopy(filtered_finish)
            
            if not finish: ### empty list
                continue #### go to next min-max gr
            else:
                warehouse_order = create_warehouse_order(finish)
                
            logger.info(f"->>> WareHouse priority {warehouse_order}")
            
            ### exclude taken_stocks out of stock_to_use
            if not taken_stocks:
                pass
            else:
                remained_stocks = {
                        s: {**s_info}
                        for s, s_info in stocks_to_use.items()
                        if s not in taken_stocks
                    }
                stocks_to_use = copy.deepcopy(remained_stocks)
            
            logger.info(f"->>> Number of stocks left: {len(stocks_to_use)}")
            
            ### SORT STOCK BY coil center priority change by customer preference
            i = 0 # i = {0, 1, 2}
            while i < 3: ### INITIATE stock from WH priority
                stocks_by_wh = filter_stocks_by_wh(stocks_to_use,[warehouse_order[i]])
                
                if len(stocks_by_wh.keys()) > 0:
                    logger.info(f"->>> cut in WH {warehouse_order[i]}")
                    coil_center_priority_cond = True
                    break
                else:
                    logger.info(f'Out of stocks for for WH {warehouse_order[i]}')
                    i +=1
                    if i == 3: coil_center_priority_cond = False
            
            logger.info(f"->>> SUB-TASK: {len(finish.keys())} FINISH  w {len(stocks_by_wh.keys())}")
            
            nx_wh_stocks = []      
            while coil_center_priority_cond: ### OPERATOR COIL CENTER to cut until all FG-in MIN MAX GR DONE
                
                if not nx_wh_stocks:
                    pass #### empty
                else: #### refresh stocks filtered by min max weight
                    stocks_by_wh = copy.deepcopy(nx_wh_stocks)
                    
                bound = 1
                while bound < 4: ### ### OPERATOR UPPER BOUND = {1, 2, 3}
                    args_dict = {
                                'logger': logger,
                                'finish': finish,
                                'stocks': stocks_by_wh,
                                'min_w': min_weight,
                                'max_w': max_weight,
                                'PARAMS': PARAMS,
                                'bound': bound,
                                'margin_df': margin_df,
                                }
                    try:
                        logger.warning("*** NORMAL case ***")
                        final_solution_patterns, over_cut, taken_stocks = multistocks_pipeline(**args_dict, prob_type="Dual")
                    except ValueError: ### ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                        if len(stocks_by_wh) == 1 and len(finish) == 1 and bound == 1: #### SEMI CASE, set bound ==3 always, bound chi loop 1 lan
                            logger.warning('*** SEMI case *** 1 FG 1 Stock, hard to optimize')
                            steel = SemiProb(stocks_by_wh, finish, PARAMS)
                            steel.update(margin_df)
                            steel.cut_n_create_new_stock_set()
                            
                            #### Update lai stock
                            stocks_to_use.pop(list(stocks_by_wh.keys())[0]) # truong hop ko cat duoc 
                            stocks_to_use.update(steel.remained_stocks)     # thi 2 dong nay bu tru nhau
                            stocks_to_use.update(steel.taken_stocks)

                            try:
                                taken_stock = list(steel.taken_stocks.keys())[0]
                                final_solution_patterns = [{'stock':        taken_stock, 
                                                            'stock_width':  steel.taken_stocks[taken_stock]['width'],
                                                            'stock_weight': steel.taken_stocks[taken_stock]['weight'],
                                                            'cuts':         steel.cut_dict, 
                                                            'trim_loss': 9999, 'trim_loss_pct': 9999
                                                         }] #SEMI TRIM LOSS lon
                            except IndexError:
                                final_solution_patterns = []
                            
                        elif len(stocks_by_wh) == 1: #### REWIND - truong hop try rewind vs 1 stock rat lon trong stocks???
                            logger.warning("*** REWIND case ***")
                            try:
                                final_solution_patterns, over_cut, taken_stocks = multistocks_pipeline(**args_dict, prob_type="Rewind")
                            except ValueError: ### empty final_sol_pattern
                                pass
                        else: pass ### ko co nghiem cat REWIND va SEMI

                    if not final_solution_patterns:
                        logger.info(f"->>>> The solution - bound {bound} is empty.")
                        bound += 1
                    else: ### neu co nghiem thi break while bound < 4 loop
                        total_taken_stocks.append(taken_stocks)
                        df = transform_to_df(final_solution_patterns)
                        cleaned_param_set = clean_filename(param_set)
                        filename = f"scr/results/solution-{cleaned_param_set}-{customer_name}-{k}-{warehouse_order[i]}.xlsx"
                        df.to_excel(filename, index=False)
                        logger.info(f"->>>> The solution {param_set}-{customer_name}-{k} bound {bound} saved")
                        break 
                
                if i < 2: # To nex COIL CENTER priority if still able
                    i +=1
                    nx_wh_stocks = filter_stocks_by_wh(stocks_to_use, [warehouse_order[i]]) #try to find STOCKS in next WH
                    try:
                        has_negative_over_cut = any(value < 0 for value in over_cut.values())
                        coil_center_priority_cond = (has_negative_over_cut & (len(nx_wh_stocks)!=0)) #can cat tiep va co coil o next priority
                        finish =  refresh_finish(finish, over_cut)                      # Left stock to cut in next WH
                    except TypeError or AttributeError:                                 # overcut empty -> chua optimized duoc o coil center n-1
                        coil_center_priority_cond = (len(nx_wh_stocks)!=0)            
                else: coil_center_priority_cond = False                                 # da het coil center de tim
            
            logger.info(f" STOCK-AFTER-CUT {customer_name}-{k}: {over_cut}")
        logger.info(f'--- DONE TASK {customer_name}---')

logger.info('**** ALL JOB ENDED **** ')