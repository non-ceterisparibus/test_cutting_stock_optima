# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import logging
import datetime
import json
import copy
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
            finish[f]['need_cut'] = over_cut[f] # finish need cut am
        else: 
            finish[f]['need_cut'] = 0
    
    return finish

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

def multistocks_cut(logger, finish, stocks, PARAMS, bound, margin_df, prob_type):
    # pipeline to cut all possible stock with finish demand upto upperbound
    steel = CuttingStocks(finish, stocks, PARAMS)
    steel.update(bound = bound, margin_df=margin_df)
    steel.filter_stocks() # flow khong consider min va max_weight
    cond = steel.set_prob(prob_type) #  "Dual" / "Rewind"
    final_solution_patterns = []
    while cond: 
        # loop to cut as many FG as possible with < 4% trim loss
        print(".")
        stt, final_solution_patterns, over_cut = steel.solve_prob()
        if not final_solution_patterns: # empty solution
            logger.warning(f"Status Infeasible")
            break
        else:
            logger.warning(f"Status {stt}")
            logger.info(f">>>> Take stock {[p['stock'] for p in final_solution_patterns]}")
            logger.info(f">>>> Overcut amount {over_cut}")
            cond = steel.check_status()
            if not cond:
                # Finalize results
                try:
                    mean_3fc = {
                        key: (
                            sum(v for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3']) if not math.isnan(v)) /
                            sum(1 for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3']) if not math.isnan(v))
                        ) if any(not math.isnan(v) for v in (finish[key]['fc1'], finish[key]['fc2'], finish[key]['fc3'])) else float('nan')
                        for key in over_cut.keys()
                    }
                    over_cut_rate = {key: round(over_cut[key]/mean_3fc[key], 4) for key in over_cut.keys()}
                    logger.info(f">>>> STOCKS USED {len(final_solution_patterns)}")
                    logger.info(f">>>> TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}")
                    logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
                except ZeroDivisionError:
                    logger.info(f">>>> STOCKS USED {len(final_solution_patterns)}")
                    logger.info(f">>>> TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}")
                    logger.warning(">>>> ZERO Forecast Data")
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
logging.basicConfig(filename=f'scr/log/test-dual-{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LOAD CONFIG & DATA
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
i = 21

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
logger.info(f'-> START processing JOB {i} PARAMS: [{batch}]')
total_taken_stocks = []
taken_stocks =[]

## Loop FINISH - each TASK (by CUSTOMER) in JOB 
for finish_item in finish_list['param_finish'][param_set]['customer']: 
    try:  ## Add try-catch for any error happen -> go to next job
        ## GET CUSTOMER-NAME -- 2/2 LAYER
        customer_name = list(finish_item.keys())[0]
        finish_by_gr = finish_item[customer_name]  ## original finish with stock ratio < 0.5 ( need cut can be positive)
        logger.info(f"->> for CUSTOMER {customer_name}")
        
        # Get beginning finish list
        og_finish = finish_item[customer_name]  # original finish with stock ratio < 0.5 (need cut can be positive)
        over_cut = {}
        final_solution_patterns = []
        
        ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
        filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
        if len(filtered_finish) < 3:
            finish = copy.deepcopy(og_finish)
        else:
            finish = copy.deepcopy(filtered_finish)
        if not finish:
            continue #### go to next min-max gr
        else:
            warehouse_order = create_warehouse_order(finish)
        logger.info(f"->>> WareHouse priority {warehouse_order}")
        logger.info(f"->>> Number of stocks left: {len(stocks_to_use)}")
        
        ### SORT STOCK BY coil center priority change by customer preference
        j = 0 # j = {0, 1, 2}
        while j < 3: ### INITIATE stock from WH priority
            stocks_by_wh = filter_stocks_by_wh(stocks_to_use,[warehouse_order[j]])
            if len(stocks_by_wh.keys()) > 0:
                logger.info(f"->>> cut in WH {warehouse_order[j]}")
                coil_center_priority_cond = True
                break
            else:
                logger.warning(f'>>>> Out of stocks for for WH {warehouse_order[j]}')
                j +=1
                if j == 3: 
                    coil_center_priority_cond = False
        logger.info(f"->>> SUB-TASK: {len(finish.keys())} FINISH  w {len(stocks_by_wh.keys())} MC")
        nx_wh_stocks = []      
        while coil_center_priority_cond: ### OPERATOR COIL CENTER to cut until all FG-in MIN MAX GR DONE
            if not nx_wh_stocks: # Enpty
                pass 
            else: 
                #### Refresh stocks
                stocks_by_wh = copy.deepcopy(nx_wh_stocks)
                logger.info(f"->>> cut in WH {warehouse_order[j]}")
            bound = 1
            print("RUNNING....\n")
            while bound < 4: ### ### OPERATOR UPPER BOUND = {1, 2, 3}
                args_dict = {
                            'logger': logger,
                            'finish': finish,
                            'stocks': stocks_by_wh,
                            'PARAMS': PARAMS,
                            'bound': bound,
                            'margin_df': margin_df,
                            }
                try:
                    logger.info("*** NORMAL case ***")
                    final_solution_patterns, over_cut, taken_stocks = multistocks_cut(**args_dict, prob_type="Dual")

                    ### Exclude taken_stocks out of stock_to_use only for dual case
                    if not taken_stocks:
                        pass # Does nothing
                    else:
                        remained_stocks = {
                                s: {**s_info}
                                for s, s_info in stocks_to_use.items()
                                if s not in taken_stocks
                            }
                        stocks_to_use = copy.deepcopy(remained_stocks)
                except ValueError: ### ko co nghiem DUAL, ko su dung stocks nao, con nguyen
                    if len(stocks_by_wh) == 1 and len(finish) == 1 and bound == 1: #### SEMI CASE, set bound ==3 always, bound chi loop 1 lan
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
                            final_solution_patterns, over_cut, taken_stocks, remained_stocks = multistocks_cut(**args_dict, prob_type="Rewind")
                            logger.info(f"REMAINED stocks: {remained_stocks}")
                            stocks_to_use.pop(list(stocks_by_wh.keys())[0]) # truong hop ko cat duoc 
                            stocks_to_use.update(remained_stocks)     # thi 2 dong nay bu tru nhau
                        except ValueError: ### empty final_sol_pattern
                            pass
                    else: 
                        pass ### NO SOLUTION for REWIND va SEMI
                # Complete Cutting in current warehouse
                if not final_solution_patterns:
                    logger.warning(f"->>>> The solution - bound {bound} is empty.")
                    bound += 0.5
                else: 
                    ### neu co nghiem thi break while bound < 4 
                    total_taken_stocks.append(taken_stocks)
                    
                    # --- SAVE to DF ---
                    df = transform_to_df(final_solution_patterns)
                    cleaned_param_set = clean_filename(param_set)
                    filename = f"scr/results/test-dual-{cleaned_param_set}-{customer_name}-{warehouse_order[j]}.xlsx"
                    df.to_excel(filename, index=False)
                
                    logger.info(f"->>>> The solution {param_set}-{customer_name}- bound {bound} EXCEL saved")
                    break 
                
            # To nex COIL CENTER priority if still able
            if j < 2: 
                j +=1
                nx_wh_stocks = filter_stocks_by_wh(stocks_to_use, [warehouse_order[j]]) #try to find STOCKS in next WH
                try:
                    has_negative_over_cut = any(value < 0 for value in over_cut.values())
                    coil_center_priority_cond = (has_negative_over_cut & (len(nx_wh_stocks)!=0)) #can cat tiep va co coil o next priority
                    finish =  refresh_finish(finish, over_cut)                      # Remained need_cut finish to cut in next WH
                    f_list = {f : (finish[f]['need_cut']) for f in finish.keys()}
                    logger.warning(f"->>> Remained finish to cut in next WH {f_list}")
                except TypeError or AttributeError:                                 # overcut empty -> chua optimized duoc o coil center n-1
                    coil_center_priority_cond = (len(nx_wh_stocks)!=0)            
            else: 
                coil_center_priority_cond = False                                 # da het coil center de tim
        logger.info(f">>> END CUTTING STOCK-AFTER-CUT for {customer_name}: {over_cut}")
   
        logger.info(f'--- DONE TASK {customer_name} ---')
    except Exception as e:
        logger.warning(f"Error with Customer {customer_name}: {type(e)} {e}")
        logger.info(f"Occured on line {traceback.extract_tb(e.__traceback__)[-1].lineno}")
        continue
        
logger.info('**** TEST JOB ENDED **** ')