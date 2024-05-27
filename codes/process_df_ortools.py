
import pandas as pd 
import numpy as np

def process_data(df,df_stock):
    df['Mother_Weight'] = df_stock['Weigth'][0]
    df['Mother_Width']= df_stock['Width'][0]
    df['Inventory_ID'] = df_stock['Inventory_ID'][0]
    df['Expected_Qt']=  df.apply(lambda df: 0 if df['Need_Cut']>=0 else round(abs(df['Remain_Weight']/(df['Width']*df_stock['Weigth'][0]/df_stock['Width'][0])) + 0.5), axis=1) #+ 0.5
    
    df['Trim_Loss'] = df_stock['Width'][0]
    df['Current_Weight'] = 0  
    df['Max_Quant']= df.apply(lambda df: 0 if df['Need_Cut']>=0 else round((df['Max_Weight']*df_stock['Width'][0])/(df_stock['Weigth'][0]*df['Width']) + 0.5 ), axis= 1) #+ 0.5        
    
    df_stock = df_stock.drop(index=0).reset_index(drop= True)
    df = df[['Order_ID','Customer', 'Maker', 'Spec_Name', 'Thickness', 'Width', 'Length', 
             'Need_Cut', 'Inventory_ID','Mother_Weight', 'Mother_Width','Max_Weight', 'Expected_Qt',
             'Num_Split','Max_Quant','Total_Quant','Trim_Loss','Current_Weight','Remain_Weight']]
    # df_nc = df[df.Need_Cut >= 0].reset_index(drop= True)
    df = df[df.Need_Cut <0].reset_index(drop= True)
    return df, df_stock

def create_data_model(df,mc_width, MIN_MARGIN):
    """Stores the data for the problem."""
    data = {}
    data['constraint_coeffs'] = [
        list(df['width'].values)
        ]
    data['width_bounds'] = [mc_width - MIN_MARGIN] 
    data['obj_coeffs'] = list(df['width'].values) 
    data['num_vars'] = len(df)
    data['num_constraints'] = len(data['constraint_coeffs'])
    return data