from removeDirtyData import saveDataset


def reflect_inflation(df_list, inflation):
    df_name_list = ["inflation_garlic_df", "inflation_cabbage_df", "inflation_radish_df", "inflation_pepper_df"]
    inflation_score = inflation[['년도', '월', '소비자물가총지수']] # we need only this 3 features

    for year in range(len(inflation_score)):
        for df in df_list:
            for i in range(len(df)):
                if df.loc[i, '연도'] == inflation_score.loc[year, '년도']:
                    if df.loc[i, '연도'] == 2023: # if '연도' == 2023, there are no total inflation score, so set each month's inflation score
                        if df.loc[i, '월'] == inflation_score.loc[year, '월']:
                            df.loc[i, '소비자물가총지수'] = inflation_score.loc[year, '소비자물가총지수']
                            # To minimize the impact of price increases due to inflation as much as possible
                            df.loc[i, '인플레이션 반영가'] = df.loc[i, '소매일일가격'] / inflation_score.loc[year, '소비자물가총지수']
                    else:
                        df.loc[i, '소비자물가총지수'] = inflation_score.loc[year, '소비자물가총지수']
                        df.loc[i, '인플레이션 반영가'] = df.loc[i, '소매일일가격'] / inflation_score.loc[year, '소비자물가총지수']

    for df, name in zip(df_list, df_name_list):
        saveDataset(df, "reflect_inflation/", name)
