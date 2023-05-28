import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def saveDataset(df, file_path, file_name):
    df.to_csv(file_path+file_name+".csv", encoding="utf-8")

def remove_outliers(df, column):
     # 이상치 제거를 위한 IQR 계산
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    # 이상치 범위 계산
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 이상치 제거
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def removeOutliers(data_list, data_name_list, columns, file_path):
    for df, df_name in zip(data_list, data_name_list):
        for column in columns:
            df = remove_outliers(df, column)
            saveDataset(df, file_path, df_name)