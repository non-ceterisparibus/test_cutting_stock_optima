# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import logging
import datetime
import json
import copy
import re

from model.O32_linear_prob import *

today = datetime.datetime.today()
formatted_date = today.strftime("%d-%m-%y")

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'cutting_one_stock_{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LOAD CONFIG & DATA
margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')

logger.info('*** LOAD OLD AND CHANGED STOCKS ***')
with open(f'scr/jobs_by_day/retry-stock-{formatted_date}.json', 'r') as stocks_file:
    retry_stock =json.load(stocks_file)

with open(f'scr/jobs_by_day/results-{formatted_date}.json', 'r') as result_file:
    result_list = json.load(result_file)
    
with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
    finish_list =json.load(finish_file)

logger.info('--- PROCEED JOBS ---')
total_solution_json = {"date": formatted_date,
                        "status":200,
                        'solution':[]
                        }

if __name__ == "__main__":
    # 1.. TAKE OLD STOCK/param set -> ko can xem coil center? chi can min margin
    split_request_id = list(retry_stock['split_request_id'].keys())[0]
    print(split_request_id)
    
    newstock = retry_stock['split_request_id'][split_request_id]['newstock']
    logger.info(f"NEW STOCK: {newstock}")
    oldstock = retry_stock['split_request_id'][split_request_id]['oldstock']
    oldstock_key = list(oldstock.keys())[0]
    
    param = retry_stock['split_request_id'][split_request_id]['param']
    param_id = param['maker']+"+"+param["spec_name"]+"+"+str(param["thickness"])
    
    # 2../ FIND SET OF FG
    finish = {}
    result = result_list[oldstock_key]
    finish_cuts = result['cuts']
    logger.info(f"OLD STOCK trim loss {result['trim_loss_pct']}")
    logger.info(f" Old result: {result['cuts']}")
    
    customer = result['customer_short_name']
    print(customer)
    finish_by_cust = finish_list['param_finish'][param_id]['customer'] # dict in array
    for finish_ls in finish_by_cust: #finish_ls dict
        cust_id = list(finish_ls.keys())[0]
        if cust_id == customer:         
            total_finish = finish_ls[cust_id]
            print(total_finish)
            finish = {key: value for key, value in total_finish.items() if key in finish_cuts}
            break
        else: pass
            
    # replace with new stocks -> optimize on trim loss only?
    steel = CuttingOneStock(finish,newstock,param)
    steel.update(margin_df)
    steel.set_prob()
    print(f"Stock: {steel.prob.stock}")
    probstt, solution, weight_cuts = steel.solve_prob()
    if probstt == "Solved":
      logger.info(F"NEW SOLUTION: {solution}")
      logger.info(weight_cuts)
    else:
        logger.warning("NO NEW SOLUTION")
    # return result
    
