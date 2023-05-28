import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

garlic_df = pd.read_csv("merged_data/garlic_merged_list.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("merged_data/napa_cabbage_merged_list.csv", low_memory=False)
radish_df = pd.read_csv("merged_data/radish_merged_list.csv", low_memory=False)
pepper_df = pd.read_csv("merged_data/pepper_merged_list.csv", low_memory=False)

garlic_df.drop("Unnamed: 0", axis=1, inplace=True)
napa_cabbage_df.drop("Unnamed: 0", axis=1, inplace=True)
radish_df.drop("Unnamed: 0", axis=1, inplace=True)
pepper_df.drop("Unnamed: 0", axis=1, inplace=True)

garlic_df.drop("Unnamed: 0.1", axis=1, inplace=True)
napa_cabbage_df.drop("Unnamed: 0.1", axis=1, inplace=True)
radish_df.drop("Unnamed: 0.1", axis=1, inplace=True)
pepper_df.drop("Unnamed: 0.1", axis=1, inplace=True)

def fibonacci(n):
    fib_list = [0, 1]
    for i in range(2, n):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list

fibonacci_number = fibonacci(100)

def modifyWithDirtyData(df, idx):
    num = random.randint(1, 5)
    
    row = df.iloc[idx]
    if num == 1:
        row = df.iloc[idx]
        row["최고기온(°C)"] = 150
    elif num == 2:
        row['품목'] = '항정살'
    elif num == 3:
        temp = row['최고기온(°C)']
        row['최고기온(°C)'] = row['최저기온(°C)']
        row['최저기온(°C)'] = temp
    elif num == 4:
        row['월'] = 13
    elif num == 5:
        if row['월'] == 2:
            row['일'] = 30
        elif row['월'] in [4,6, 9, 11]:
            row['일'] == 31
        else:
            row['일'] == 32
    df.iloc[idx] = row

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]

for df, df_name in zip(df_list, df_name_list):
    df_length = len(df)
    for num in fibonacci_number:
        if num >= df_length:
            num %= df_length
            modifyWithDirtyData(df, num)
    print(df)
    df.to_csv("add_dirtydata/"+df_name+".csv", encoding="utf-8")
