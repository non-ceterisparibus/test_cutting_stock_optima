
import numpy as np
import pandas as pd


def get_po_data(df_po,Thickness,Spec,stock_name): 
    data= df_po
    con1 = data['Need_Cut'] < 0
    con2 = data['Stock_Name'] == stock_name
    con3 = data['Thickness'] == Thickness
    # con4 = data['Cutting_Note'].isna()
    con5 = data['Spec_Name'] == Spec
    # con6 = data['Customer'] == Customer
    # con7 = data['Length'] == Length
    
    data[['Thickness','Need_Cut']] = data[['Thickness','Need_Cut']].astype('float64')
    data['Need_Cut']= round(data['Need_Cut'])

    tmp= data[con2 & con3 & con5].sort_values(by= ['Need_Cut', 'Width'], ascending= [True, True]).reset_index(drop= True) #& con6
    tmp['Order_ID'] = tmp.index
    return tmp

def get_stock_data(df_stock,Thickness,Spec,sort = [True]):
    st_con1= df_stock['Spec_Name'] == Spec
    st_con2= df_stock['Thickness'] == Thickness
    # st_con3= df_stock['Remark'].isin(['Ok',np.nan])
    # st_con4= df_stock['Cus_Number'].isna()
    # st_con5= df_stock['Width'] == Width
    tmp_stock = df_stock[st_con1 & st_con2 ] #& st_con5
    tmp_st = tmp_stock[['Inventory_ID','Spec_Name','Thickness','Width','Length','Weigth']].sort_values(by= ['Weigth'], ascending= sort).reset_index(drop=True)
    # tmp_stock['Receipt_Date'] = tmp_stock['Receipt_Date'].astype('datetime64[ns]')
    # tmp_stock['HTV_Note'] = tmp_stock['HTV_Note'].replace(0, np.nan)
    # tmp_st = tmp_stock[tmp_stock['HTV_Note'].isna()].sort_values(by= ['Weigth','Receipt_Date'], ascending= [False, True]).reset_index(drop=True)[['Inventory_ID','Spec_Name','Thickness','Width','Length','Quantity','Weigth','Receipt_Date']]
    # tmp_st = tmp_stock.sort_values(by= ['Weigth','Receipt_Date'], ascending= [False, True]).reset_index(drop=True)[['Inventory_ID','Spec_Name','Thickness','Width','Length','Quantity','Weigth','Receipt_Date']]
    return tmp_st

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

def sel_extra_coil(df, df_1):
    results = 0
    # df_1.Expected_Qt = 2
    df_1 = df_1[~df_1.Order_ID.isin(df.Order_ID.unique())]
    if len(df_1) >0:
        for i in ['Inventory_ID', 'Mother_Weight', 'Mother_Width']:
            df_1[i] = df[i][0]
        df_1[['Expected_Qt', 'Trim_Loss', 'Current_Weight','Run_Time(ms)']] = 0
        df_1['Max_Quant'] = 2
        # df_1[['Inventory_ID', 'Mother_Weight', 'Mother_Width', 'Expected_Qt', 'Max_Quant', 'Trim_Loss', 'Current_Weight']] = 0
        df_1 = df_1[list(df.columns)]

        df_tmp = pd.DataFrame(columns= df.columns)
        # data = pd.DataFrame(columns= df.columns)
        for i in range(len(df_1)):
            data= pd.concat([df,df_1[i:i+1]]).reset_index(drop=True)
            try:
                # print(df_1[i:i+1].Order_ID.unique())
                Num_Split,Trim_Loss,Run_Time = optimize_model_v4(data,Width,min_loss,max_loss, True)
            except:
                Trim_Loss = df['Mother_Width'][0]
            if i == 0:
                results = Trim_Loss
                df_tmp = data
            if Trim_Loss < results:
                results = Trim_Loss
                df_tmp = data
        return df_tmp
    else:
        return df

