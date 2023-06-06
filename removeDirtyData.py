import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_valid_date(year, month, day):
    # Check if it's a leap year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # Maximum days for each month
    max_days = [31, 28 + is_leap_year, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Check valid year, month, and day range
    if 1 <= month <= 12 and 1 <= day <= max_days[month - 1]:
        return True
    else:
        return False
    
def removeDirtyData(df, item):
    # 1: Remove rows where average relative humidity exceeds 100%
    df = df[df['평균 상대습도(%)'] <= 100]

    # 2: Remove rows where minimum temperature is greater than maximum temperature
    df = df[df['최저기온(°C)'] <= df['최고기온(°C)']]

    # 3: Remove rows where both export and import are zero
    df = df[(df['수출(kg)'] != 0) | (df['수입(kg)'] != 0)]

    # 4: Remove rows where item is different
    df = df[df['품목'] == item]

    # 5: Remove rows where year is before 2024
    df = df[df['연도'] < 2024]
    
    # 6: Remove invalid dates
    df = df[df.apply(lambda row: is_valid_date(row['연도'], row['월'], row['일']), axis=1)]

    # 7: Remove rows where any temperature data is below -20°C or above 45°C
    df = df[(df['평균기온(°C)'] > -20) & (df['평균기온(°C)'] < 45)]
    df = df[(df['최저기온(°C)'] > -20) & (df['최저기온(°C)'] < 45)]
    df = df[(df['최고기온(°C)'] > -20) & (df['최고기온(°C)'] < 45)]
    df = df[(df['평균 지면온도(°C)'] > -20) & (df['평균 지면온도(°C)'] < 45)]
    
    # 8: Remove rows where minimum relative humidity exceeds 100%
    df = df[df['최소 상대습도(%)'] <= 100]
    
    return df

def saveDataset(df, file_path, file_name):
    df.to_csv(file_path + file_name + ".csv", encoding="utf-8")

# Remove Dirty Data and Save Dataset
def remove_save(data_list, data_name_list, item_list, file_path):
    for df, item, df_name in zip(data_list, item_list, data_name_list):
        df = removeDirtyData(df, item)
        saveDataset(df, file_path, df_name)
    return data_list

def plot_comparison(data_before, data_after, data_name):
    plt.figure(figsize=(12, 6))
    
    # Visualize data before removing dirty data
    plt.subplot(1, 2, 1)
    plt.title(f"{data_name} - Before Removing Dirty Data - 평균 상대습도(%)")
    plt.hist(data_before["평균 상대습도(%)"], bins=20)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # Visualize data after removing dirty data
    plt.subplot(1, 2, 2)
    plt.title(f"{data_name} - After Removing Dirty Data")
    plt.hist(data_after["평균 상대습도(%)"], bins=20)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Visualize data before and after removing dirty data
def compare_dirty_data(data_list_before, data_list_after, data_name_list):
    for data_before, data_after, data_name in zip(data_list_before, data_list_after, data_name_list):
        plot_comparison(data_before, data_after, data_name)
        
