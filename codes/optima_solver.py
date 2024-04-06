
from process_data import create_data_model
from process_data import process_data
import pandas as pd 
import numpy as np
import warnings
import time
import math
from ortools.linear_solver import pywraplp

def create_data_model(df,Width,min_loss,max_loss):

    """
    Create Obj Function
    and 
    Constraints
    """
    data = {}
    data['constraint_coeffs'] = [
        list(df['Width'].values),
        # list(-df['Width'].values),
        ]
    data['bounds'] = [Width - min_loss] # Width - min_loss, 
    data['obj_coeffs'] = list(df['Width'].values) 
    data['num_vars'] = len(df)
    data['num_constraints'] = len(data['constraint_coeffs'])
    return data


def main_v1(df,df_stock,process_type = 'Default',percent = 1,max_w_1= True,max_w_2=True):
    start = time.time()
    df['Num_Split'] = 0
    if process_type == 'Default':
        df['Max_Weight'] = df.apply(lambda df : 0 if df['Need_Cut'] >= 0 else abs(df['Need_Cut']),1)
    elif process_type == 'Auto':
        df['Max_Weight'] = abs(df['Need_Cut'])*round(percent,2)
    else:
        df['Max_Weight'] = 0
        for or_id, max_weight in process_type: 
            df.at[df[df.Order_ID == or_id].index[0], 'Max_Weight'] = max_weight

    df['Total_Quant'] = 0
    df['Remain_Weight'] = df['Need_Cut']

    df_nc = df.copy()
    df_nc[['Inventory_ID', 'Mother_Weight', 'Mother_Width', 'Expected_Qt', 'Max_Quant', 'Trim_Loss', 'Current_Weight']] = 0
    df,df_stock = process_data(df, df_stock)
    final_output = pd.DataFrame(columns= df.columns)
    
    Width= df['Mother_Width'][0]
    Thickness= df['Thickness'][0]

    if Thickness <= 3 and Thickness > 1:
        min_loss = 8
    elif Thickness == 1:
        min_loss = 5.5
    else:
        min_loss = 3*Thickness      
    max_loss= int(0.04*Width)   
    try:
        i=0
        while len(df) > 0:
            print(f'Lần cắt thứ {i+1}')
            Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df,Width,min_loss,max_loss, max_w_1)
            
            if round(Trim_Loss,2) > round(min_loss,2):    #max_loss
                df_op = df.copy()
                df_op_final =  df.copy()
                df_op_final['Num_Split'],df_op_final['Trim_Loss'],df_op_final['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                cur_loss = Trim_Loss #len(df) + Trim_Loss
                for lop in range(len(df_nc)):
                    # print(cur_loss)
                    df_op = sel_extra_coil(df_op, df_nc)
                    try:
                        Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df_op,Width,min_loss,max_loss, max_w_2)
                        df_op['Num_Split'],df_op['Trim_Loss'],df_op['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                        df_check = df_op[df_op.Num_Split < df_op.Expected_Qt].reset_index(drop=True)
                        if len(df_check) == 0 and Trim_Loss <= max_loss:
                            check_point = Trim_Loss/10
                        else:
                            check_point = Trim_Loss     #len(df_check) + Trim_Loss
                        # check_point = len(df_check) + Trim_Loss
                        print('Optimizing Trim Loss ...... ',Trim_Loss)
                        # if int(check_point) <= int(min_loss):
                        #     df_op_final = df_op
                        #     break
                        if len(df_check) ==0 and int(Trim_Loss) == min_loss:  #<= int(max_loss/2)
                            df_op_final = df_op
                            break
                        if check_point < cur_loss:
                            cur_loss = check_point
                            df_op_final = df_op
                    except:
                        pass
                df = df_op_final    
            else:
                df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                
            df['Total_Quant'] += df['Num_Split']
            df['Current_Weight'] = (df['Num_Split']*df.Width*df['Mother_Weight'])/df['Mother_Width']
            df['Remain_Weight'] += df['Current_Weight']
            df['Max_Weight'] = df.apply(lambda df: df['Max_Weight'] - df['Current_Weight'] if df['Max_Weight'] != 0 else df['Max_Weight'],1)
            df['Trim_Loss_Weight'] = df.apply(lambda df: (df['Trim_Loss']*df['Mother_Weight'])/df['Mother_Width'],1)

            final_output= pd.concat([final_output,df[df.Num_Split > 0]]).reset_index(drop=True)
            df = df[df.Num_Split < df.Expected_Qt].reset_index(drop=True)

            i += 1
            # print(final_output)
            if len(df) != 0 :
                try:
                    df,df_stock = process_data(df, df_stock)
                    print(f'Số Coil còn lại: {len(df)}')
                except:
                    print(f'Trong kho có {i} MC ... Số MC này không đủ để cắt !!!! ')
                    print(f'Số Coil còn lại: {len(df)}')
                    break
            if i > 10:
                break
    except:
        print('Ăn lol rồi !!!')
        i=0
        while len(df) > 0:
            print(f'Lần cắt thứ {i+1}')
            Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df,Width,min_loss,max_loss, max_w_1)
            
            if Trim_Loss > max_loss:    #max_loss
                df_op = df.copy()
                df_op_final =  df.copy()
                cur_loss = len(df) + Trim_Loss
                for lop in range(len(df_nc)):
                    # print(cur_loss)
                    df_op = sel_extra_coil(df_op, df_nc)
                    try:
                        Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df_op,Width,min_loss,max_loss, max_w_2)
                        df_op['Num_Split'],df_op['Trim_Loss'],df_op['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                        df_check = df_op[df_op.Num_Split < df_op.Expected_Qt].reset_index(drop=True)
                        if len(df_check) == 0 and Trim_Loss <= max_loss:
                            check_point = Trim_Loss/10
                        else:
                            check_point = len(df_check) + Trim_Loss
                        # check_point = len(df_check) + Trim_Loss
                        print('Optimizing Trim Loss ...... ',Trim_Loss)
                        # if int(check_point) <= int(min_loss):
                        #     df_op_final = df_op
                        #     break
                        if len(df_check) ==0 and int(Trim_Loss) == min_loss:  #<= int(max_loss/2)
                            df_op_final = df_op
                            break
                        if check_point < cur_loss:
                            cur_loss = check_point
                            df_op_final = df_op
                    except:
                        pass
                df = df_op_final    
            else:
                df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                
            df['Total_Quant'] += df['Num_Split']
            df['Current_Weight'] = (df['Num_Split']*df.Width*df['Mother_Weight'])/df['Mother_Width']
            df['Remain_Weight'] += df['Current_Weight']
            df['Max_Weight'] = df.apply(lambda df: df['Max_Weight'] - df['Current_Weight'] if df['Max_Weight'] != 0 else df['Max_Weight'],1)
            df['Trim_Loss_Weight'] = df.apply(lambda df: (df['Trim_Loss']*df['Mother_Weight'])/df['Mother_Width'],1)

            final_output= pd.concat([final_output,df[df.Num_Split > 0]]).reset_index(drop=True)
            df = df[df.Num_Split < df.Expected_Qt].reset_index(drop=True)

            i += 1
            # print(final_output)
            if len(df) != 0 :
                try:
                    df,df_stock = process_data(df, df_stock)
                    print(f'Số Coil còn lại: {len(df)}')
                except:
                    print(f'Trong kho có {i} MC ... Số MC này không đủ để cắt !!!! ')
                    print(f'Số Coil còn lại: {len(df)}')
                    break
            if i > 10:
                break
            
    final_output['Total_MC']= i
    final_output['Total_Remain']= sum(final_output['Remain_Weight'][final_output['Remain_Weight'] >=0 ])
    final_output['Total_Trim_Loss_Weight'] = float(final_output.groupby(['Inventory_ID'])['Trim_Loss_Weight'].unique().reset_index().Trim_Loss_Weight.sum())
    final_output['Total_Run_Time(s)'] = time.time() - start
    #float(final_output.groupby(['Inventory_ID'])['Run_Time(ms)'].unique().reset_index()['Run_Time(ms)'].sum())
    return final_output,df

def main_v2(df,df_stock,process_type = 'Default',percent = 1,max_w_1= True,max_w_2=True):
    start = time.time()
    df['Num_Split'] = 0
    if process_type == 'Default':
        df['Max_Weight'] = df.apply(lambda df : 0 if df['Need_Cut'] >= 0 else abs(df['Need_Cut']),1)

    df['Total_Quant'] = 0
    df['Remain_Weight'] = df['Need_Cut']

    df_nc = df.copy()
    df_nc[['Inventory_ID', 'Mother_Weight', 'Mother_Width', 'Expected_Qt', 'Max_Quant', 'Trim_Loss', 'Current_Weight']] = 0

    df,df_stock = process_data(df, df_stock)
    final_output = pd.DataFrame(columns= df.columns)
    df_check = df.copy()

    Width= df['Mother_Width'][0]
    Thickness= df['Thickness'][0]
    df_nc = df_nc[df.columns]

    if Thickness <= 3 and Thickness > 1:
        min_loss = 8     #8
    elif Thickness == 1:
        min_loss = 5.5
    else:
        min_loss = 3*Thickness      
    max_loss= int(0.04*Width)   
    i=0
###################################################
    while len(df) > 0:
        print(f'Lần cắt thứ {i+1}')
        Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df,Width,min_loss,max_loss, max_w_1)
        # print(Trim_Loss)

        if sum(df.Expected_Qt) == sum(Num_Split) and Trim_Loss < max_loss:
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            
        elif sum(Num_Split) == sum(df.Expected_Qt) and Trim_Loss >= max_loss:
            # print('2')
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            df_nc_new = df_nc.copy()
            Width_new = Trim_Loss
            Num_Split_new,Trim_Loss_new,Run_Time_new = optimize_model_v4(df_nc_new,Width_new,min_loss,max_loss, False)
            df_nc_new['Num_Split'],df_nc_new['Trim_Loss'],df_nc_new['Run_Time(ms)'] = Num_Split_new,Trim_Loss_new,Run_Time_new
            for cols in ['Inventory_ID', 'Mother_Weight', 'Mother_Width']:
                df_nc_new[cols] = df[cols][0]

            df = pd.concat([df, df_nc_new[df_nc_new.Num_Split > 0]])
            df['Trim_Loss'] = Trim_Loss_new
            df['Run_Time(ms)'] += Run_Time_new
            # print(Trim_Loss_new)
            
        # elif round(Trim_Loss,2) > round(min_loss,2):
        #     print('3')
        #     df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
        #     df_nc_new = df_nc.copy()
        #     Width_new = Trim_Loss
        #     Num_Split_new,Trim_Loss_new,Run_Time_new = optimize_model_v4(df_nc_new,Width_new,min_loss,max_loss, False)
        #     df_nc_new['Num_Split'],df_nc_new['Trim_Loss'],df_nc_new['Run_Time(ms)'] = Num_Split_new,Trim_Loss_new,Run_Time_new
        #     for cols in ['Inventory_ID', 'Mother_Weight', 'Mother_Width']:
        #         df_nc_new[cols] = df[cols][0]

        #     df = pd.concat([df, df_nc_new[df_nc_new.Num_Split > 0]])
        #     df['Trim_Loss'] = Trim_Loss_new
        #     df['Run_Time(ms)'] += Run_Time_new
        #     print(Trim_Loss_new)
                
        elif round(Trim_Loss,2) > round(min_loss,2) : 
            df_op = df.copy()
            df_op_final =  df.copy()
            df_op_final['Num_Split'],df_op_final['Trim_Loss'],df_op_final['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            cur_loss = len(df) + Trim_Loss
            for lop in range(len(df_nc)):
                # print(cur_loss)
                df_op = sel_extra_coil(df_op, df_nc)
                try:
                    Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df_op,Width,min_loss,max_loss, max_w_2)
                    df_op['Num_Split'],df_op['Trim_Loss'],df_op['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
                    df_check = df_op[df_op.Num_Split < df_op.Expected_Qt].reset_index(drop=True)
                    if len(df_check) == 0 and Trim_Loss <= max_loss:
                        check_point = Trim_Loss/10
                    else:
                        check_point = len(df_check) + Trim_Loss
                    print('Optimizing Trim Loss ...... ',Trim_Loss)

                    if len(df_check) ==0 and int(Trim_Loss) == min_loss:  #<= int(max_loss/2)
                        df_op_final = df_op
                        break
                    if check_point < cur_loss:
                        cur_loss = check_point
                        df_op_final = df_op
                except:
                    pass
            df = df_op_final  

        else:
            # print(4)
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time

        df['Total_Quant'] += df['Num_Split']
        df['Current_Weight'] = (df['Num_Split']*df.Width*df['Mother_Weight'])/df['Mother_Width']
        df['Remain_Weight'] += df['Current_Weight']
        df['Max_Weight'] = df.apply(lambda df: df['Max_Weight'] - df['Current_Weight'] if df['Max_Weight'] != 0 else df['Max_Weight'],1)
        df['Trim_Loss_Weight'] = df.apply(lambda df: (df['Trim_Loss']*df['Mother_Weight'])/df['Mother_Width'],1)

        final_output= pd.concat([final_output,df[df.Num_Split > 0]]).reset_index(drop=True)
        df = df[df.Num_Split < df.Expected_Qt].reset_index(drop=True)

        i += 1
        # print(final_output) 
        if len(df) != 0 :
            try:
                df,df_stock = process_data(df, df_stock)
                print(f'Số Coil còn lại: {len(df)}')
            except:
                print(f'Trong kho có {i} MC ... Số MC này không đủ để cắt !!!! ')
                print(f'Số Coil còn lại: {len(df)}')
                break
        if i > 15:
            break

        print('So coil con lai: ', len(df))
###################################################

    final_output['Total_MC']= i
    final_output['Total_Remain']= sum(final_output['Remain_Weight'][final_output['Remain_Weight'] >=0 ])
    final_output['Total_Trim_Loss_Weight'] = float(final_output.groupby(['Inventory_ID'])['Trim_Loss_Weight'].unique().reset_index().Trim_Loss_Weight.sum())
    # final_output['Total_Run_Time(s)'] = time.time() - start
    #float(final_output.groupby(['Inventory_ID'])['Run_Time(ms)'].unique().reset_index()['Run_Time(ms)'].sum())
    return final_output,df

def optimize_model_v4(df,Width,min_loss,max_loss, max_quant = False):     
    data = create_data_model(df,Width,min_loss,max_loss)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP') #GLOP  SCIP  PDLP Clp  GLPK  CBC  XPRESS SAT

    if not solver:
        return

    # infinity = solver.infinity()
    x = {}
    if max_quant == False:
        for j in range(data['num_vars']):
            x[j] = solver.IntVar(0, solver.infinity(), 'x[%i]' % j) #int(tmp_['Max_Quant'][j])
    else:
        for j in range(data['num_vars']):
            x[j] = solver.IntVar(0, int(df['Max_Quant'][j]), 'x[%i]' % j) #int(tmp_['Max_Quant'][j])
    # print('Number of variables =', solver.NumVariables())

    for i in range(data['num_constraints']):
     constraint_expr = \
    [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
     solver.Add(sum(constraint_expr) <= data['bounds'][i])
    
    obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
    solver.Minimize((Width - min_loss)  - solver.Sum(obj_expr)) #(Width - min_loss)
    
    status = solver.Solve()
    # print(solver.Objective().Gap())
    # print(status)
    if status == pywraplp.Solver.OPTIMAL:
        sol_values = []
        # print('Objective value =', solver.Objective().Value())
        for j in range(data['num_vars']):
            sol_values.append(x[j].solution_value())
            # print(x[j].name(), ' = ', x[j].solution_value())
        # print(solver.Objective().BestBound())
        # print('Problem solved in %f milliseconds' % solver.wall_time())
        # print('Problem solved in %d iterations' % solver.iterations())
        # print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
        # return sol_values , Width - solver.Objective().Value(), solver.wall_time()
    else:
        print('The problem does not have an optimal solution.')
        # return 0 , Width, solver.wall_time()
    # return sol_values , Width - solver.Objective().Value(), solver.wall_time()
    trim_loss = min_loss + solver.Objective().Value() #min_loss + 
    # trim_loss = solver.Objective().Value()
    # trim_loss = max_loss - solver.Objective().Value()
    
    return sol_values ,trim_loss, solver.wall_time()

def main_v3(df,df_stock,process_type = 'Default',percent = 1,max_w_1= True,max_w_2=True):
    start = time.time()
    df['Num_Split'] = 0
    if process_type == 'Default':
        df['Max_Weight'] = df.apply(lambda df : 0 if df['Need_Cut'] >= 0 else abs(df['Need_Cut']),1)

    df['Total_Quant'] = 0
    df['Remain_Weight'] = df['Need_Cut']

    df_nc = df.copy()
    df_nc[['Inventory_ID', 'Mother_Weight', 'Mother_Width', 'Expected_Qt', 'Max_Quant', 'Trim_Loss', 'Current_Weight']] = 0

    df,df_stock = process_data(df, df_stock)
    final_output = pd.DataFrame(columns= df.columns)
    df_check = df.copy()

    Width= df['Mother_Width'][0]
    Thickness= df['Thickness'][0]
    df_nc = df_nc[df.columns]

    if Thickness <= 3 and Thickness > 1:
        min_loss = 8     #8
    elif Thickness == 1:
        min_loss = 5.5
    else:
        min_loss = 3*Thickness      
    max_loss= int(0.04*Width)   
    i=0
###################################################
    while len(df) > 0:
        print(f'Lần cắt thứ {i+1}')
        Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df,Width,min_loss,max_loss, max_w_1)
        # print(Trim_Loss)

        if sum(df.Expected_Qt) == sum(Num_Split) and Trim_Loss < max_loss:
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            
        elif sum(Num_Split) == sum(df.Expected_Qt) and Trim_Loss >= max_loss:
            # print('2')
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            df_nc_new = df_nc.copy()
            Width_new = Trim_Loss
            Num_Split_new,Trim_Loss_new,Run_Time_new = optimize_model_v4(df_nc_new,Width_new,min_loss,max_loss, False)
            df_nc_new['Num_Split'],df_nc_new['Trim_Loss'],df_nc_new['Run_Time(ms)'] = Num_Split_new,Trim_Loss_new,Run_Time_new
            for cols in ['Inventory_ID', 'Mother_Weight', 'Mother_Width']:
                df_nc_new[cols] = df[cols][0]

            df = pd.concat([df, df_nc_new[df_nc_new.Num_Split > 0]])
            df['Trim_Loss'] = Trim_Loss_new
            df['Run_Time(ms)'] += Run_Time_new
            # print(Trim_Loss_new)
            
        elif round(Trim_Loss,2) > round(min_loss,2):
            print('3')
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
            df_nc_new = df_nc.copy()
            Width_new = Trim_Loss
            Num_Split_new,Trim_Loss_new,Run_Time_new = optimize_model_v4(df_nc_new,Width_new,min_loss,max_loss, False)
            df_nc_new['Num_Split'],df_nc_new['Trim_Loss'],df_nc_new['Run_Time(ms)'] = Num_Split_new,Trim_Loss_new,Run_Time_new
            for cols in ['Inventory_ID', 'Mother_Weight', 'Mother_Width']:
                df_nc_new[cols] = df[cols][0]

            df = pd.concat([df, df_nc_new[df_nc_new.Num_Split > 0]])
            df['Trim_Loss'] = Trim_Loss_new
            df['Run_Time(ms)'] += Run_Time_new
            # print(Trim_Loss_new)
                
        # elif round(Trim_Loss,2) > round(min_loss,2) : 
        #     df_op = df.copy()
        #     df_op_final =  df.copy()
        #     df_op_final['Num_Split'],df_op_final['Trim_Loss'],df_op_final['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
        #     cur_loss = len(df) + Trim_Loss
        #     for lop in range(len(df_nc)):
        #         # print(cur_loss)
        #         df_op = sel_extra_coil(df_op, df_nc)
        #         try:
        #             Num_Split,Trim_Loss,Run_Time = optimize_model_v4(df_op,Width,min_loss,max_loss, max_w_2)
        #             df_op['Num_Split'],df_op['Trim_Loss'],df_op['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time
        #             df_check = df_op[df_op.Num_Split < df_op.Expected_Qt].reset_index(drop=True)
        #             if len(df_check) == 0 and Trim_Loss <= max_loss:
        #                 check_point = Trim_Loss/10
        #             else:
        #                 check_point = len(df_check) + Trim_Loss
        #             print('Optimizing Trim Loss ...... ',Trim_Loss)

        #             if len(df_check) ==0 and int(Trim_Loss) == min_loss:  #<= int(max_loss/2)
        #                 df_op_final = df_op
        #                 break
        #             if check_point < cur_loss:
        #                 cur_loss = check_point
        #                 df_op_final = df_op
        #         except:
        #             pass
        #     df = df_op_final  

        else:
            # print(4)
            df['Num_Split'],df['Trim_Loss'],df['Run_Time(ms)'] = Num_Split,Trim_Loss,Run_Time

        df['Total_Quant'] += df['Num_Split']
        df['Current_Weight'] = (df['Num_Split']*df.Width*df['Mother_Weight'])/df['Mother_Width']
        df['Remain_Weight'] += df['Current_Weight']
        df['Max_Weight'] = df.apply(lambda df: df['Max_Weight'] - df['Current_Weight'] if df['Max_Weight'] != 0 else df['Max_Weight'],1)
        df['Trim_Loss_Weight'] = df.apply(lambda df: (df['Trim_Loss']*df['Mother_Weight'])/df['Mother_Width'],1)

        final_output= pd.concat([final_output,df[df.Num_Split > 0]]).reset_index(drop=True)
        df = df[df.Num_Split < df.Expected_Qt].reset_index(drop=True)

        i += 1
        # print(final_output) 
        if len(df) != 0 :
            try:
                df,df_stock = process_data(df, df_stock)
                print(f'Số Coil còn lại: {len(df)}')
            except:
                print(f'Trong kho có {i} MC ... Số MC này không đủ để cắt !!!! ')
                print(f'Số Coil còn lại: {len(df)}')
                break
        if i > 15:
            break

        print('So coil con lai: ', len(df))
###################################################

    final_output['Total_MC']= i
    final_output['Total_Remain']= sum(final_output['Remain_Weight'][final_output['Remain_Weight'] >=0 ])
    final_output['Total_Trim_Loss_Weight'] = float(final_output.groupby(['Inventory_ID'])['Trim_Loss_Weight'].unique().reset_index().Trim_Loss_Weight.sum())
    # final_output['Total_Run_Time(s)'] = time.time() - start
    #float(final_output.groupby(['Inventory_ID'])['Run_Time(ms)'].unique().reset_index()['Run_Time(ms)'].sum())
    return final_output,df