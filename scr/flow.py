import pandas as pd
import logging
import datetime
import json
from dual_solver import Cuttingtocks

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

# LOAD JOB-LIST
with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
    job_list = json.load(file)

with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'r') as stocks_file:
    stocks_list =json.load(stocks_file)

with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
    finish_list = json.load(finish_file)

n_jobs = job_list['number of job']

for i in range(n_jobs):
    bound = 2

    # LOAD
    param_set = job_list['jobs'][i]['param']
    stocks_available = job_list['jobs'][i]['stocks_available']
    tasks = job_list['jobs'][i]['tasks']

    total_need_cut = sum(item['total_need_cut'] for item in tasks.values())
    PARAMS = stocks_list['param_finish'][param_set]['param']
    batch = PARAMS['code']

    #### CONVERT F-S TO DICT 
    stocks = stocks_list['param_finish'][param_set]['stocks']
    for finish_item in finish_list['param_finish'][param_set]['customer']:
        customer = list(finish_item.keys())[0]
        print(customer)
        og_finish = finish_item[customer]

        ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
        filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
        if len(filtered_finish) < 3:
            finish = og_finish
        else:
            finish = filtered_finish

        # START
        logger.info(f'START processing job {i} PARAMS: [{batch}] with over_cut bound {bound} forecast month')
        logger.info(f'Process {len(finish.keys())} FINISH  w {len(stocks.keys())} stocks for CUSTOMER {customer}')

        #### Calculating the sum of 'total_need_cut' values
        if stocks_available >= 6 * total_need_cut:
            logger.warning(" May over cut too much ")
        elif total_need_cut < stocks_available <= 3 * total_need_cut:
            logger.info("May have enough stocks to cut")
        elif stocks_available < total_need_cut:
            logger.warning("Lacks of stocks")