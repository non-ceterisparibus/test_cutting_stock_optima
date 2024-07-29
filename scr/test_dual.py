import pandas as pd
import logging
import datetime
import json
from model.O42_cutting_stocks import Cuttingtocks
# from model.O31_steel_objects import FinishObjects, StockObjects

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
today = datetime.datetime.today() # Format the date to dd/mm/yy
formatted_date = today.strftime("%d-%m-%y")

margin_df = pd.read_csv('scr/model_config/min_margin.csv')
spec_type = pd.read_csv('scr/model_config/spec_type.csv')
# coil_priority = pd.read_csv('scr/data/coil_data.csv')

# START
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'cutting_stocks_{formatted_date}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LOAD JOB-LIST
# with open(f'scr/jobs_by_day/job-list-{formatted_date}.json', 'r') as file:
#     job_list = json.load(file)

# with open(f'scr/jobs_by_day/stocks-list-{formatted_date}.json', 'r') as stocks_file:
#     stocks_list =json.load(stocks_file)

# with open(f'scr/jobs_by_day/finish-list-{formatted_date}.json', 'r') as finish_file:
#     finish_list = json.load(finish_file)

# SET UP
# i= 13
# bound = 2 # ----> operator to change bound dynamically

# # PROCESS
# param_set = job_list['jobs'][i]['param']
# stocks_available = job_list['jobs'][i]['stocks_available']
# tasks = job_list['jobs'][i]['tasks']

# total_need_cut = sum(item['total_need_cut'] for item in tasks.values())
# PARAMS = stocks_list['param_finish'][param_set]['param']
# batch = PARAMS['code']

# #### CONVERT F-S TO DICT - c
# stocks = stocks_list['param_finish'][param_set]['stocks']
# finish_item = finish_list['param_finish'][param_set]['customer'][0]
# customer = list(finish_item.keys())[0]
# og_finish = finish_item[customer]

# ### THEM DIEU KIEN FILTER FINISH: NEU NUMBER NEED CUT < 0 IT QUA THI MOI LAY CA KO CAN CAT
# filtered_finish = {k: v for k, v in og_finish.items() if v['need_cut'] < 0}
# if len(filtered_finish) < 3:
#     finish = og_finish
# else:
#     finish = filtered_finish
param_set = "POSCOVN+SPCC-SD+1.0"
PARAMS = {
            "spec_name": "SPCC-SD",
            "type": "Carbon",
            "thickness": 1.0,
            "maker": "POSCOVN",}
stocks = {
            "TP238H002948-1": {
               "receiving_date": 45088,
               "width": 1219,
               "weight": 3000,
               "warehouse": "HSC"
            },
            "TP238H002948-2": {
               "receiving_date": 45088,
               "width": 1219,
               "weight": 3459,
               "warehouse": "HSC"
            }
         }
finish = {
                  "F200": {
                     "customer_name": "CIC",
                     "width": 210.0,
                     "need_cut": -908.876582278481,
                     "fc1": 2256.2464,
                     "fc2": 1984.6524,
                     "fc3": 1934.0102
                  },
                  "F197": {
                     "customer_name": "CIC",
                     "width": 150.0,
                     "need_cut": -300.0,
                     "fc1": 731.842155,
                     "fc2": 687.165744375,
                     "fc3": 577.24759125
                  },
                  "F190": {
                     "customer_name": "CIC",
                     "width": 100.0,
                     "need_cut": -300.0,
                     "fc1": 486.09712,
                     "fc2": 812.15629,
                     "fc3": 714.013235
                  },
                  "F198": {
                     "customer_name": "CIC",
                     "width": 152.0,
                     "need_cut": -178.0,
                     "fc1": 625.8,
                     "fc2": 983.4,
                     "fc3": 596.0
                  },
                  "F203": {
                     "customer_name": "CIC",
                     "width": 60.0,
                     "need_cut": -109.0,
                     "fc1": 1373.2299846,
                     "fc2": 1331.7674778,
                     "fc3": 905.3461206
                  },
                  "F196": {
                     "customer_name": "CIC",
                     "width": 125.0,
                     "need_cut": -100.0,
                     "fc1": 138.35134375,
                     "fc2": 130.7123125,
                     "fc3": 48.38053125
                  },
}
batch = "POSCOVN SPCC-SD 1.0"
bound = 1
customer = "CIC"
# START
logger.info(f'START processing job {"_"} PARAMS: [{batch}] with over_cut bound {bound} forecast month')
logger.info(f'Process {len(finish.keys())} FINISH  w {len(stocks.keys())} stocks for CUSTOMER {customer}')

# #### Calculating the sum of 'total_need_cut' values
# if stocks_available >= 6 * total_need_cut:
#     logger.warning(" May over cut too much ")
# elif total_need_cut < stocks_available <= 3 * total_need_cut:
#     logger.info("May have enough stocks to cut")
# elif stocks_available < total_need_cut:
#     logger.warning("Lacks of stocks")

# RUN OPTIMA
cond = True
steel = Cuttingtocks(finish, stocks, PARAMS)
steel.update(bound = bound, margin_df=margin_df)
steel.set_dualprob()
while cond == True:
    stt, final_solution_patterns, over_cut = steel.solve_dualprob()
    try:
        logger.info(f'> Take stock {[p['stock'] for p in final_solution_patterns]}')
        logger.info(f'>> Overcut amount {over_cut}')
        cond = steel.check_status()
        if not cond:
            over_cut_rate = {key: round(over_cut[key]/finish[key]['fc1'], 4) for key in over_cut.keys()}
            logger.info(f">>> Total stock used {len(final_solution_patterns)}")
            logger.info(f'TRIM LOSS PERCENT OF EACH STOCK {[p['trim_loss_pct'] for p in final_solution_patterns]}')        
            logger.info(f">> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
        else:
            steel.refresh_data()
            logger.info(f">>> Stocks to continue to cut {steel.dualprob.dual_stocks.keys()}")

    except TypeError:
        logger.info(f"Status {stt}")
        break

# tranform solution to pandas and save file named: solution-{batch}-{customer}-{date}
if not final_solution_patterns:
    logger.info(f"The solution - bound {bound} is empty.")
    # bound += 1
else:
    df = transform_to_df(final_solution_patterns)
    filename = f"scr/results/solution-{param_set}-{customer}.xlsx"
    df.to_excel(filename, index=False)
    logger.info("saved to csv")


logger.info(f"-->> SOLUTION {final_solution_patterns}")
logger.info('ENDED')