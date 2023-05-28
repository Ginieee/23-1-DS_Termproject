import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# 1. 데이터셋 로드
# ==============================================

# ----------------------------------------------
# 이지해 데이터셋(소매가격 + 기상관측)
# ----------------------------------------------
# 배추, 무, 마늘, 건고추
# 
# Feature) 
# 일시
# 품목
# 소매일일가격
# 평균기온(°C)
# 최저기온(°C)
# 최고기온(°C)
# 최소 상대습도(%)
# 평균 상대습도(%)
# 최대 풍속(m/s)
# 평균 풍속(m/s)
# 합계 일사량(MJ/m2)
# 합계 일조시간(hr)
# 평균 지면온도(°C)
# ----------------------------------------------
napa_cabbage_plus_assos_df = pd.read_csv("weather/napa_cabbage_plus_assos_df.csv", low_memory=False)
radish_plus_assos_df = pd.read_csv("weather/radish_plus_assos_df.csv", low_memory=False)
galric_plus_assos_df = pd.read_csv("weather/garlic_plus_assos_df.csv", low_memory=False)
pepper_plus_assos_df = pd.read_csv("weather/pepper_plus_assos_df.csv", low_memory=False)

assos_df_list = [napa_cabbage_plus_assos_df, radish_plus_assos_df, galric_plus_assos_df, pepper_plus_assos_df]

for df in assos_df_list:
    df['일시'] = pd.to_datetime(df['일시'], format="%Y-%m-%d")

    # 연도, 월, 일 추출
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day

# ----------------------------------------------
# 강어진 데이터셋(소매가격)
# ----------------------------------------------
napa_cabbage_plus_price_df = pd.read_csv("price/df_cabbage.csv", low_memory=False, encoding='cp949')
radish_plus_price_df = pd.read_csv("price/df_radish.csv", low_memory=False, encoding='cp949')
galric_plus_price_df = pd.read_csv("price/df_garlic.csv", low_memory=False, encoding='cp949')
pepper_plus_price_df = pd.read_csv("price/df_pepper.csv", low_memory=False, encoding='cp949')

print(napa_cabbage_plus_price_df)

# ----------------------------------------------
# 장원준 데이터셋(소매가격 + 수입,수출)
# ----------------------------------------------
napa_cabbage_plus_income_export_df = pd.read_excel("income/income_export_cabbage.xlsx")
radish_plus_income_export_df = pd.read_excel("income/income_export_radish.xlsx")
galric_plus_income_export_df = pd.read_excel("income/income_export_garlic.xlsx")
pepper_plus_income_export_df = pd.read_excel("income/income_export_pepper.xlsx")

income_export_df_list = [napa_cabbage_plus_income_export_df, radish_plus_income_export_df, galric_plus_income_export_df, pepper_plus_income_export_df]

for df in income_export_df_list:
    df = df.rename(columns={'year': '연도', 'month': '월', 'export(kg)': "수출(kg)", "export($)": "수출(달러)", "income(kg)":"수입(kg)", "income($)":"수입(달러)"}, inplace=True)


# ==============================================
# 2. 데이터셋 합치기
# ==============================================

# ----------------------------------------------
# 이지해 데이터셋 + 장원준 데이터셋
# ----------------------------------------------
merged_df_name_list = ['napa_cabbage_merged_list', "radish_merged_list", "garlic_merged_list", "pepper_merged_list"]
for assos_df, income_df, df_name in zip(assos_df_list, income_export_df_list, merged_df_name_list):
    merged_df = pd.merge(assos_df, income_df, on=['연도', '월'], how='inner')
    # merged_df.drop(axis=1, inplace=True)
    # napa_cabbage_plus_assos_df.drop(napa_cabbage_plus_assos_df[napa_cabbage_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
    merged_df.to_csv("merged_data/"+df_name+".csv", encoding="utf-8")
    print(merged_df)

