import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def reflact_inflation(df_list, inflation):
    infletion_score = inflation[['년도', '소비자물가총지수']]
    for year in range(len(infletion_score)):
        for df in df_list:
            for i in range(len(df)):
                if df.loc[i,'연도'] == infletion_score.loc[year,'년도']:
                    df.loc[i,'인플레이션 반영가'] = df.loc[i,'소매일일가격'] / infletion_score.loc[year,'소비자물가총지수']