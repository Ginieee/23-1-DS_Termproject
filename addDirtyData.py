import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def setting(data_list):
    # Remove unnecessary columns from the data
    for data in data_list:
        data.drop("Unnamed: 0", axis=1, inplace=True)
        data.drop("Unnamed: 0.1", axis=1, inplace=True)

def fibonacci(n):
    # Generate a list of Fibonacci numbers up to the given index
    fib_list = [0, 1]
    for i in range(2, n):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list

fibonacci_number = fibonacci(100)

def modifyWithDirtyData(df, idx):
    # Modify a random row in the DataFrame with dirty data
    num = random.randint(1, 5)
    
    row = df.iloc[idx]
    if num == 1:
        # Set an extremely high temperature value
        row["최고기온(°C)"] = 150
    elif num == 2:
        # Change the value in the '품목' column
        row['품목'] = '항정살'
    elif num == 3:
        # Swap the values of '최고기온(°C)' and '최저기온(°C)'
        temp = row['최고기온(°C)']
        row['최고기온(°C)'] = row['최저기온(°C)']
        row['최저기온(°C)'] = temp
    elif num == 4:
        # Set an invalid value for the '월' column
        row['월'] = 13
    elif num == 5:
        # Set an invalid value for the '일' column based on the month
        if row['월'] == 2:
            row['일'] = 30
        elif row['월'] in [4, 6, 9, 11]:
            row['일'] = 31
        else:
            row['일'] = 32
    df.iloc[idx] = row

def addDirtyData(data_list, data_name_list):
    # Add dirty data to the given data list
    for df, df_name in zip(data_list, data_name_list):
        df_length = len(df)
        for num in fibonacci_number:
            if num >= df_length:
                num %= df_length
            modifyWithDirtyData(df, num)
        # Save the modified DataFrame as a CSV file
        df.to_csv("add_dirtydata/"+df_name+".csv", encoding="utf-8")