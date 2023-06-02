from removeDirtyData import saveDataset


def reflect_inflation(df_list, inflation):
    df_name_list = ["final_garlic_df", "final_cabbage_df", "final_radish_df", "final_pepper_df"]
    inflation_score = inflation[['년도', '소비자물가총지수']]
    for year in range(len(inflation_score)):
        for df in df_list:
            for i in range(len(df)):
                if df.loc[i, '연도'] == inflation_score.loc[year, '년도']:
                    df.loc[i, '인플레이션 반영가'] = df.loc[i, '소매일일가격'] / inflation_score.loc[year, '소비자물가총지수']

    for df, name in zip(df_list, df_name_list):
        saveDataset(df, "reflect_inflation/", name)
