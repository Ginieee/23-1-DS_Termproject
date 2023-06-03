import numpy as np
import pandas as pd

#1. date타입 변경
#2. 1999년 데이터 버림
#3. 11~12월 데이터만 남김
#4. 가격별로 상, 중, 하 나눠서 feature 추가
#5. 완료한 데이터 csv파일로 넘김

def final_df_classification(df_list, name_list):
    
    for df, name in zip(df_list, name_list):
        df['일시'] = pd.to_datetime(df['일시'])
        
        #1999년 데이터 버림
        df['일시'] = pd.to_datetime(df['일시'])
        df = df[df['일시'].dt.year != 1999]
        
        #11~12월 데이터만 남김
        df = df[(df['일시'].dt.month == 11) | (df['일시'].dt.month == 12)]
        
        #가격별로 상, 중, 하 나눠서 feature 추가
        point1 = df['인플레이션 반영가'].min() + (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min())/3
        point2 = df['인플레이션 반영가'].min() + 2 * (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min())/3
        
        #새로운 feature 초기화
        df['가격'] = -1
        
        #for idx in range(df.length)
        # point1보다 작은 값은 1으로 설정
        df.loc[df['인플레이션 반영가'] < point1, '가격'] = 1

        # point1보다 크고 point2보다 작은 값은 2로 설정
        df.loc[(df['인플레이션 반영가'] >= point1) & (df['인플레이션 반영가'] < point2), '가격'] = 2

        # point2보다 큰 값은 3으로 설정
        df.loc[df['인플레이션 반영가'] >= point2, '가격'] = 3
        
        df.to_csv("final_df_classification/"+name+"_df.csv", encoding="utf-8")