# GET OVERLAP PARAMS

import pandas as pd
import numpy as np
import math

# INPUT
fin_file_path = "../data/20240710_finish_df.xlsx"
mc_file_path = "../data/20240710_mc_df.xlsx"

# PROCESS
fin_df = pd.read_excel(fin_file_path)
mc_df = pd.read_excel(mc_file_path)

has_need_cut_df = fin_df[fin_df['need_cut'] < -10]
has_need_cut_df['params']= has_need_cut_df['maker'] + "+" + has_need_cut_df['spec_name']+ "+" + has_need_cut_df['thickness'].astype(str)
fin_params = has_need_cut_df['params'].unique()

mc_df['params'] =  mc_df['maker'] + "+" + mc_df['spec_name']+ "+" + mc_df['thickness'].astype(str)
mc_params = mc_df['params'].unique()

# Find the intersection (overlapping values)
overlap = set(mc_params) & set(fin_params)

# Convert the result back to a list if needed
params = list(overlap)
n_jobs = len(params)

# OUTPUT
# n_jobs
# params

# Save params as list of job to run
# params