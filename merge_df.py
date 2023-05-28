import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#extract date of weather dataset
def extract_datetime(data_list):
    for df in data_list:
        df['일시'] = pd.to_datetime(df['일시'], format="%Y-%m-%d")

        # 연도, 월, 일 추출
        df['연도'] = df['일시'].dt.year
        df['월'] = df['일시'].dt.month
        df['일'] = df['일시'].dt.day

#rename reatures of income_export dataset
def rename_features(data_list):
    for df in data_list:
        df = df.rename(columns={'year': '연도', 'month': '월', 'export(kg)': "수출(kg)", "export($)": "수출(달러)", "income(kg)":"수입(kg)", "income($)":"수입(달러)"}, inplace=True)

#merge dataset
def merge_dataset(data_list1, data_list2):
    merged_df_name_list = ['napa_cabbage_merged_list', "radish_merged_list", "garlic_merged_list", "pepper_merged_list"]
    for assos_df, income_df, df_name in zip(data_list1, data_list2, merged_df_name_list):
        merged_df = pd.merge(assos_df, income_df, on=['연도', '월'], how='inner')
        # merged_df.drop(axis=1, inplace=True)
        # napa_cabbage_plus_assos_df.drop(napa_cabbage_plus_assos_df[napa_cabbage_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
        merged_df.to_csv("merged_data/"+df_name+".csv", encoding="utf-8")
        #print_merged_dataset(merged_df)

def print_merged_dataset(dataset):
    print(dataset)