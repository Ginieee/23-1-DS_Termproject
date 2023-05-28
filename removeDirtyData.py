import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_valid_date(year, month, day):
    # 윤년인지 확인
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # 월별 최대 일 수
    max_days = [31, 28 + is_leap_year, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 유효한 연도, 월, 일 범위 확인
    if 1 <= month <= 12 and 1 <= day <= max_days[month - 1]:
        return True
    else:
        return False
    
def removeDirtyData(df, item):
    # 1: 평균 상대습도가 100%를 초과하는 경우 해당 row 제거
    df = df[df['평균 상대습도(%)'] <= 100]

    # 2: 최저기온과 최고기온이 역전된 경우 해당 row 제거
    df = df[df['최저기온(°C)'] <= df['최고기온(°C)']]

    # 3: 수출(kg)과 수입(kg)가 모두 0인 경우 해당 row 제거
    df = df[(df['수출(kg)'] != 0) | (df['수입(kg)'] != 0)]

    # 4: 품목이 다른 경우 해당 row 제거
    df = df[df['품목'] == item]

    # 5: 연도가 2022 이전인 경우 해당 row 제거
    df = df[df['연도'] < 2024]
    
    # 6: 유효하지 않은 일자 삭제
    df = df[df.apply(lambda row: is_valid_date(row['연도'], row['월'], row['일']), axis=1)]
    
    return df

def saveDataset(df, file_path, file_name):
    df.to_csv(file_path+file_name+".csv", encoding="utf-8")

if __name__ == "__main__":
    # Read files
    garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
    napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
    radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
    pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

    df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
    df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
    item_list = ['마늘', '배추', '무', '건고추']    
    file_path = "remove_dirtyData/"

    # Remove Dirty Data and Save Dataset
    for df, item, df_name in zip(df_list, item_list, df_name_list):
        df = removeDirtyData(df, item)
        saveDataset(df, file_path, df_name)
