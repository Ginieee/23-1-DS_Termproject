import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

plt.rcParams['font.family'] = 'Malgun Gothic'

napa_cabbage_df = pd.read_csv("merged_data/napa_cabbage_merged_list.csv", low_memory=False, index_col=0)
radish_df = pd.read_csv("merged_data/radish_merged_list.csv", low_memory=False, index_col=0)
garlic_df = pd.read_csv("merged_data/garlic_merged_list.csv", low_memory=False, index_col=0)
pepper_df = pd.read_csv("merged_data/pepper_merged_list.csv", low_memory=False, index_col=0)


df_list = [napa_cabbage_df, radish_df, garlic_df, pepper_df]
for df in df_list:
    df.drop(df[df['소매일일가격'] == 0].index, inplace=True)
    df.drop('Unnamed: 0.1', inplace=True, axis=1)
    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균기온(°C)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 상대습도(%)']])

    df['평균기온+평균상대습도'] = df['A_normalized'] + df['B_normalized']

    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균기온(°C)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 풍속(m/s)']])

    df['평균기온+평균풍속'] = df['A_normalized'] + df['B_normalized']

    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균기온(°C)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 상대습도(%)']])

    df['평균기온+평균상대습도'] = df['A_normalized'] + df['B_normalized']

    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균기온+평균상대습도']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 풍속(m/s)']])

    df['평균기온+상대습도+풍속'] = df['A_normalized'] + df['B_normalized']


    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균 풍속(m/s)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['합계 일사량(MJ/m2)']])

    df['평균풍속+합계일사'] = df['A_normalized'] + df['B_normalized']


    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균 풍속(m/s)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 지면온도(°C)']])
    
    df['평균풍속+지면온도'] = df['A_normalized'] + df['B_normalized']

    scaler = MinMaxScaler()
    df['A_normalized'] = scaler.fit_transform(df[['평균 풍속(m/s)']])
    scaler = MinMaxScaler()
    df['B_normalized'] = scaler.fit_transform(df[['평균 지면온도(°C)']])

    df.drop('A_normalized', axis=1, inplace=True)
    df.drop('B_normalized', axis=1, inplace=True)



# ==============================================
# 4. feature들 간의 상관관계 계산
# ==============================================
# 1) 배추
napa_corr = napa_cabbage_df.corr()
sns.heatmap(napa_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 2) 무
radish_corr = radish_df.corr()
sns.heatmap(radish_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 3) 마늘
garlic_corr = garlic_df.corr()
sns.heatmap(garlic_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 4) 건고추
pepper_corr = pepper_df.corr()
sns.heatmap(pepper_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()
