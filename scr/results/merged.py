import pandas as pd
import os

# Set the directory containing the Excel files
folder_path = 'scr/results'

# Get a list of all Excel files in the folder
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# Initialize an empty list to hold dataframes
df_list = []

# Loop through each Excel file and read it into a dataframe
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    df_list.append(df)

# Concatenate all dataframes into one
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged dataframe to a new Excel file
merged_df.to_excel(os.path.join(folder_path, 'merged_file_20240805.xlsx'), index=False)
