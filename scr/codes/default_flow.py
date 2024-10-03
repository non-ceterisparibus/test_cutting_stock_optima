# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import logging
import datetime
import json
import os
import copy
from model.O42_cutting_stocks import Cuttingtocks
from model.O41_semi_prob import SemiProb

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

def sort_stocks_by_wh(data, warehouse_order):
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

# LOAD CONFIG & DATA
today = datetime.datetime.today()
# Format the date to dd/mm/yy
formatted_date = today.strftime("%d-%m-%y")

margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'cutting_stocks_{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    # LOAD JOB INFO
    param_set = job_list['jobs'][i]['param']
    stocks_available = job_list['jobs'][i]['stocks_available']
    tasks = job_list['jobs'][i]['tasks']
    PARAMS = stocks_list['param_finish'][param_set]['param']
    batch = PARAMS['code']

    #### CONVERT F-S TO DICT 
    og_stocks = stocks_list['param_finish'][param_set]['stocks'] # original stocks available - phan biet vs stocks se dung cho tung customer
    stocks = copy.deepcopy(og_stocks)
    
    # START
    logger.info(f'--> START processing JOB {i} PARAMS: [{batch}]')
    
    for finish_item in finish_list['param_finish'][param_set]['customer']: # loop for each TASK IN JOB by CUSTOMER
        # reset
        over_cut = {}
        final_solution_patterns = []
        
        # get customer
        customer_name = list(finish_item.keys())[0]
        og_finish = finish_item[customer_name]  # original finish with stock ratio < 0.5 ( need cut can be positive)
        
        ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
        filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
        if len(filtered_finish) < 3:
            finish = copy.deepcopy(og_finish)
        else:
            finish = copy.deepcopy(filtered_finish)
        
        logger.info(f">> TASK: {len(finish.keys())} FINISH  w {len(stocks.keys())} stocks for CUSTOMER {customer_name}")
                
        # SORT STOCK BY coil center priority change by customer preference
        warehouse_order = create_warehouse_order(finish)
        logger.info(f"WH order {warehouse_order}")
        
        i = 0 # 1st coil center priority i = {0, 1, 2}
        while i < 3: # Initiate stock from WH priority
            stocks = sort_stocks_by_wh(stocks, [warehouse_order[i]])
            if len(stocks.keys()) > 0:
                logger.info(f">> cut in WH {warehouse_order[i]}")
                coil_center_priority_cond = True
                break
            else:
                logger.info(f'Out of stocks for for WH {warehouse_order[i]}')
                i +=1
                if i == 3: coil_center_priority_cond = False
            
        while coil_center_priority_cond: #OPERATOR COIL CENTER TO CONTROL STOCKS to cut until all FG DONE
            
            def multistocks_pipeline(finish, stocks, PARAMS, bound, margin_df, flow):
                # pipeline to cut all possible stock with finish demand upto upperbound
                cond = True
                steel = Cuttingtocks(finish, stocks, PARAMS)
                steel.update(bound = bound, margin_df=margin_df)
                steel.set_prob(flow) # set condition to set flow = "Dual" / "Rewind"
                while cond == True: 
                    # loop to cut as many FG as possible with < 4% trim loss
                    stt, final_solution_patterns, over_cut = steel.solve_prob()
                    if not final_solution_patterns: # empty solution
                        logger.warning(f"Status Infeasible")
                        break
                    else:
                        logger.warning(f"Status {stt}")
                        logger.info(f'> Take stock {[p['stock'] for p in final_solution_patterns]}')
                        logger.info(f'>> Overcut amount {over_cut}')
                        cond = steel.check_status()
                        if not cond:
                            over_cut_rate = {key: round(over_cut[key]/finish[key]['fc1'], 4) for key in over_cut.keys()}
                            logger.info(f">>> Total stock USED {len(final_solution_patterns)}")
                            logger.info(f'>>> TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}')        
                            logger.info(f">>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
                        else:
                            steel.refresh_data()
                            logger.info(f">>> Stocks to continue to cut {steel.prob.dual_stocks.keys()}")

                # UPDATE REMAINED STOCKS IF CONTINUE
                if not final_solution_patterns:
                    # Handle the case when final_solution_patterns is empty
                    taken_stocks = []
                    raise ValueError("final_solution_patterns is empty")
                else:
                    taken_stocks = [p['stock'] for p in final_solution_patterns]
                
                # Count remained stocks after cut
                remained_stocks = {
                    s: {**s_info}
                    for s, s_info in stocks.items()
                    if s not in taken_stocks
                }
                return final_solution_patterns, over_cut, remained_stocks

            bound = 1
            while bound < 4: # OPERATOR UPPER BOUND TO CONTROL - BOUND = {1, 2, 3}
                logger.info(f'--> with over_cut bound {bound} forecast month')
                try:
                    final_solution_patterns, over_cut, remained_stocks = multistocks_pipeline(finish, stocks, PARAMS, bound, margin_df, flow="Dual")
                    stocks = copy.deepcopy(remained_stocks)
                except ValueError: # ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                    if len(stocks) == 1 and len(finish) == 1:
                        logger.warning('SEMI case ->> 1 FG 1 Stock, hard to optimize')
                        steel = SemiProb(stocks, finish, PARAMS)
                        steel.update(margin_df)
                        steel.cut_n_create_new_stock_set()
                        final_solution_patterns = steel.cut_dict
                        remained_stocks = steel.remained_stocks
                    elif len(stocks) == 1: # try Rewind - truong hop try rewind vs 1 stock rat lon trong stocks???
                        logger.warning("REWIND case ->>")
                        final_solution_patterns, over_cut, remained_stocks = multistocks_pipeline(finish, stocks, PARAMS, bound, margin_df, flow="Rewind")
                        stocks = copy.deepcopy(remained_stocks)
                    else: pass #ko co nghiem cat REWIND va SEMI

                if not final_solution_patterns:
                    logger.info(f"The solution - bound {bound} is empty.")
                    bound += 1
                else:
                    # df = transform_to_df(final_solution_patterns)
                    # filename = f"scr/results/solution-{param_set}-{customer_name}-{warehouse_order[i]}.xlsx"
                    # df.to_excel(filename, index=False)
                    logger.info("saved to csv")
                    break
            
            if i < 2: # Go to next CC priority if still able
                i +=1
                stocks = sort_stocks_by_wh(stocks, [warehouse_order[i]]) #TAKE STOCKS in next WH
                try:
                    finish =  refresh_finish(finish, over_cut) # Left stock to cut in next WH
                    has_negative_over_cut = any(value < 0 for value in over_cut.values())
                    coil_center_priority_cond = (has_negative_over_cut & (len(stocks)!=0)) #can cat tiep va co coil o next priority
                except TypeError or AttributeError: # overcut empty -> chua cat duoc o coil center n-1
                    coil_center_priority_cond = (len(stocks)!=0)
            else: coil_center_priority_cond = False # khong di tim coil o center # nua
        
        logger.info(f" STOCK-AFTER-CUT customer {customer_name}: {over_cut}")
        logger.info(f'--- DONE TASK {customer_name}---')

logger.info('**** ALL JOB ENDED **** ')