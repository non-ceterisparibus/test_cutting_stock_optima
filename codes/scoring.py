
import pandas as pd 
import numpy as np
import warnings
import time
import math

from process_data import get_po_data, get_stock_data
from optima_solver import main_v1, main_v2, main_v3, optimize_model_v4

def score_(df_final_4,df_4,max_loss):
    status_4 = -1
    if len(df_final_4) > 0 and len(df_4) == 0 and max(df_final_4['Trim_Loss']) <= max_loss:  
        status_4 = 1
        print(f"""
        Total_MC: {df_final_4.Total_MC[0]}
        Total_Trim_Loss_Weight: {df_final_4.Total_Trim_Loss_Weight[0]}
        Total_Remain: {df_final_4.Total_Remain[0]}     
        Remain Order: {len(df_4)} 
        """)
        # score_4 = math.sqrt(math.pow(df_final_4.Total_MC[0],2) + 2*math.pow(df_final_4.Total_Trim_Loss_Weight[0],2) + math.pow(df_final_4.Total_Remain[0],2))
        # score_4 = math.sqrt(math.pow(df_final_4.Total_Trim_Loss_Weight[0],2) + math.pow(df_final_4.Total_Remain[0],2))
        score_4 = df_final_4.Total_Trim_Loss_Weight[0]
    else:
        score_4 = float('inf')
    return status_4,score_4, df_final_4.Total_Remain[0]

def output_table(df_po,df_st,Thickness,Spec,stock_name,Width = 1219):
    start = time.time()
    max_loss= int(0.04*Width) 
    po_data_raw = get_po_data(df_po,Thickness,Spec,stock_name)
    stock_data_raw = get_stock_data(df_st,Thickness,Spec)
    stock_data_raw4 = get_stock_data(df_st,Thickness,Spec,sort = [False])
    

    dfx = po_data_raw.reset_index(drop=True)
    df_stockx = stock_data_raw.reset_index(drop=True)
    df_stockx4 = stock_data_raw4.reset_index(drop=True)
    status= []
    score = []
    mc_ = []
    # -------------------------------------V2.1--------------------------------------------------
    df_final_1, df_1 = main_v2(dfx,df_stockx,process_type = 'Default',percent = 1,max_w_1= True,max_w_2=True)
    status_1,score_1,mc_1 = score_(df_final_1,df_1,max_loss)
    status.append(status_1)
    score.append(score_1)
    mc_.append(mc_1)
        
    df_final_2, df_2 = main_v2(dfx,df_stockx,process_type = 'Default',percent = 1,max_w_1= False,max_w_2=True)
    status_2,score_2,mc_2 = score_(df_final_2,df_2,max_loss)   
    status.append(status_2)
    score.append(score_2)
    mc_.append(mc_2)

    # -------------------------------------V2.2--------------------------------------------------
    df_final_3, df_3 = main_v2(dfx,df_stockx4,process_type = 'Default',percent = 1,max_w_1= True,max_w_2=True)
    status_3,score_3,mc_3 = score_(df_final_3,df_3,max_loss)
    status.append(status_3)
    score.append(score_3)
    mc_.append(mc_3)
    
    df_final_4, df_4 = main_v2(dfx,df_stockx4,process_type = 'Default',percent = 1,max_w_1= False,max_w_2=True)
    status_4,score_4,mc_4 = score_(df_final_4,df_4,max_loss)   
    status.append(status_4)
    score.append(score_4)
    mc_.append(mc_4)
    
    # -------------------------------------V3.1--------------------------------------------------
    df_final_5, df_5 = main_v3(dfx,df_stockx,process_type = 'Default',percent = 1,max_w_1= True)
    status_5,score_5,mc_5 = score_(df_final_5,df_5,max_loss)
    status.append(status_5)
    score.append(score_5)
    mc_.append(mc_5)
    
    df_final_6, df_6 = main_v3(dfx,df_stockx,process_type = 'Default',percent = 1,max_w_1= False)
    status_6,score_6,mc_6 = score_(df_final_6,df_6,max_loss) 
    status.append(status_6)
    score.append(score_6)
    mc_.append(mc_6)
    
    # -------------------------------------V3.2--------------------------------------------------   
    df_final_7, df_7 = main_v3(dfx,df_stockx4,process_type = 'Default',percent = 1,max_w_1= True)
    status_7,score_7,mc_7 = score_(df_final_7,df_7,max_loss)
    status.append(status_7)
    score.append(score_7)    
    mc_.append(mc_7)
    
    df_final_8, df_8 = main_v3(dfx,df_stockx4,process_type = 'Default',percent = 1,max_w_1= False)
    status_8,score_8,mc_8 = score_(df_final_8,df_8,max_loss)
    status.append(status_8)
    score.append(score_8)
    mc_.append(mc_8)

    # for i in range(len(status)):
    #     print(f'status {i+1} = {status[i]} and status_2 = {status_2} and status_2 = {status_3} and status_4 = {status_4}')
    #     print(f'score_1 = {score_1} and score_2 = {score_2} and score_3 = {score_3} and score_4 = {score_4}' )
    # # print('')
    
    print('Status: ', status)
    print('Score: ', score)
    print('Remain W: ',mc_)

    # min_ = min(score)
    # print(min_)
    chec_ = 0
    final_df = []
    inx = 0
    h = 0
    min_ = 0
    if 1 not in status:
        print('>>>>>>> Cant Find the Optimal Solution !!!! <<<<<<<')
        return
    
    for x,y,z in zip(status,score,mc_):
        if x == -1:
            pass
        else:
            if min_ == 0 and chec_ == 0:
                min_ = y
                chec_ = z
            elif y < min_:
                # if chec_ == 0:
                #     chec_ = z
                # elif z < chec_:
                inx = h
                min_ = y
        h += 1
                
    print('inx =', inx)
    
    if inx == 0:
        final_df = df_final_1
    elif inx == 1: 
        final_df = df_final_2
    elif inx == 2:
        final_df = df_final_3
    elif inx == 3:
        final_df = df_final_4
    elif inx == 4:
        final_df = df_final_5
    elif inx == 5:
        final_df = df_final_6
    elif inx == 6:
        final_df = df_final_7
    elif inx == 7:
        final_df = df_final_8 
    
    final_df['Total_Run_Time(s)'] = time.time() - start 
    # print('Total Run Time.....',time.time() - start)
    print('score = ',min_)
    return final_df