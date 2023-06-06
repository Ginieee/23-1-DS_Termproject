import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def saveDataset(df, file_path, file_name):
    df.to_csv(file_path + file_name + ".csv", encoding="utf-8")

def remove_outliers(df, column):
    # Calculate IQR for outlier removal
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    # Calculate outlier range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Remove outliers
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def removeOutliers(data_list, data_name_list, columns, file_path):
    data_after_list = []
    for df, df_name in zip(data_list, data_name_list):
        for column in columns:
            df = remove_outliers(df, column)
            saveDataset(df, file_path, df_name)
        data_after_list.append(df)
    return data_after_list

def visualize_outliers(data_before, data_after, column):
    plt.figure(figsize=(12, 6))
    
    # Visualize data before outlier removal
    plt.subplot(1, 2, 1)
    plt.title(f"Outliers - Before Removal ({column})")
    plt.boxplot(data_before[column])
    plt.ylabel("Value")
    
    # Visualize data after outlier removal
    plt.subplot(1, 2, 2)
    plt.title(f"Outliers - After Removal ({column})")
    plt.boxplot(data_after[column])
    plt.ylabel("Value")
    
    plt.tight_layout()
    plt.show()

# Compare and visualize data before and after outlier removal
def compare_outliers(data_list_before, data_list_after, column_list):
    for data_before, data_after, column in zip(data_list_before, data_list_after, column_list):
        visualize_outliers(data_before, data_after, column)
