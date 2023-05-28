import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'

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

if __name__ == "__main__":
    # Read files
    garlic_df = pd.read_csv("remove_dirtyData/garlic_df.csv", low_memory=False)
    napa_cabbage_df = pd.read_csv("remove_dirtyData/napa_cabbage_df.csv", low_memory=False)
    radish_df = pd.read_csv("remove_dirtyData/radish_df.csv", low_memory=False)
    pepper_df = pd.read_csv("remove_dirtyData/pepper_df.csv", low_memory=False)

    df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
    df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
    
    # Outlier 제거
    columns = ["평균기온(°C)","최저기온(°C)","최고기온(°C)","최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)","수출(kg)","수출(달러)","수입(kg)","수입(달러)"]
    file_path = "remove_outlier/"
    
    for df, df_name in zip(df_list, df_name_list):
        for column in columns:
            df = remove_outliers(df, column)
            saveDataset(df, file_path, df_name)
