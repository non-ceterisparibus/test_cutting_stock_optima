import pandas as pd
import logging
import datetime
# from prefect import task, flow
import json
import os
import copy
from model.O41_dual_solver import DualProblem
from model.O42_cutting_stocks import Cuttingtocks

# HELPERS
# data = [{'count': 1, 'stock': 'HTV0876/23', 'cuts': {'F364': 1, 'F361': 2, 'F360': 0, 'F365': 6, 'F359': 4, 'F367': 1}, 'trim_loss': 6.0, 'trim_loss_pct': 0.492},
#         {'count': 1, 'stock': 'TP236H005510', 'cuts': {'F351': 1, 'F349': 4, 'F347': 0, 'F348': 1, 'F354': 0, 'F357': 1}, 'trim_loss': 8.0, 'trim_loss_pct': 0.656,},
#         {'count': 1, 'stock': 'TR242H00000035', 'cuts': {'F351': 2, 'F349': 0, 'F347': 3, 'F348': 0, 'F354': 2, 'F357': 0}, 'trim_loss': 36.0, 'trim_loss_pct': 2.953, }
#         ]

# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES

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

margin_df = pd.read_csv('scr/data/min_margin.csv')
spec_type = pd.read_csv('scr/data/spec_type.csv')
# coil_priority = pd.read_csv('scr/data/coil_data.csv')

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
for i in range(n_jobs): # loop for each job
    # LOAD
    param_set = job_list['jobs'][i]['param']
    stocks_available = job_list['jobs'][i]['stocks_available']
    tasks = job_list['jobs'][i]['tasks']

    total_need_cut = sum(item['total_need_cut'] for item in tasks.values())
    PARAMS = stocks_list['param_finish'][param_set]['param']
    batch = PARAMS['code']

    #### CONVERT F-S TO DICT 
    og_stocks = stocks_list['param_finish'][param_set]['stocks'] # original stocks available
    stocks = copy.deepcopy(og_stocks)
    
    # START
    logger.info(f'--> START processing job {i} PARAMS: [{batch}]')
    
    for finish_item in finish_list['param_finish'][param_set]['customer']: # loop for each job - customer
        customer = list(finish_item.keys())[0]
        og_finish = finish_item[customer]  # original finish with stock ratio < 0.5 ( need cut can be positive)

        ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
        filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
        if len(filtered_finish) < 3:
            finish = copy.deepcopy(og_finish)
        else:
            finish = copy.deepcopy(filtered_finish)
             
        if len(stocks.keys()) == 0:
            logger.info(f'Out of stocks for for CUSTOMER {customer}')
            break
        else: logger.info(f'>> TASK {len(finish.keys())} FINISH  w {len(stocks.keys())} stocks for CUSTOMER {customer}')

        #### Calculating the sum of 'total_need_cut' values
        if stocks_available >= 6 * total_need_cut:
            logger.warning("May over cut too much ")
        elif total_need_cut < stocks_available <= 3 * total_need_cut:
            logger.warning("May have enough stocks to cut")
        elif stocks_available < total_need_cut:
            logger.warning("Lacks of stocks")
            
        # def rewind_flow(): try if and only if pattern suit width not weight?, how to define the ratio to cut
        
        def dual_flow(finish, stocks, PARAMS, bound, margin_df):
            # RUN OPTIMA
            cond = True
            steel = Cuttingtocks(finish, stocks, PARAMS)
            steel.update(bound = bound, margin_df=margin_df)
            steel.set_dualprob()
            while cond == True: # loop to cut as many FG as possible with < 4% trim loss
                stt, final_solution_patterns, over_cut = steel.solve_dualprob()
                if not final_solution_patterns: # empty solution
                    logger.warning(f"Status {stt}")
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
                        logger.info(f">>> Stocks to continue to cut {steel.dualprob.dual_stocks.keys()}")
                
            return final_solution_patterns, over_cut
        
        # UPPER BOUND - OVER CUT CONTROL - BOUND < 3
        bound = 1
        while bound < 4:
            logger.info(f'--> with over_cut bound {bound} forecast month')
            final_solution_patterns, over_cut = dual_flow(finish, stocks, PARAMS, bound, margin_df)
            try:
                # UPDATE REMAINED STOCKS IF CONTINUE
                taken_stocks = [p['stock'] for p in final_solution_patterns]
                remained_stocks = {
                    s: {**s_info}
                    for s, s_info in stocks.items()
                    if s not in taken_stocks
                }
                stocks = copy.deepcopy(remained_stocks)
            except TypeError: pass # ko co nghiem, ko su dung stocks nao, con nguyen

            # tranform solution to pandas and save file named: solution-{batch}-{customer}-{date}
            if not final_solution_patterns:
                logger.info(f"The solution - bound {bound} is empty.")
                bound += 1
            else:
                # df = transform_to_df(final_solution_patterns)
                # filename = f"scr/results/solution-{param_set}-{customer}.xlsx"
                # df.to_excel(filename, index=False)
                logger.info("saved to csv")
                break
                  
        logger.info(f" Stock after cut: {over_cut}")
        logger.info('--- DONE TASK ---')

logger.info('**** ENDED **** ')