import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setting2(data_list):
    for data in data_list:
        data.drop("Unnamed: 0", axis=1, inplace=True)
        data.drop("Unnamed: 0.1", axis=1, inplace=True)
        data.drop("Unnamed: 0.2", axis=1, inplace=True)

def draw_corr_heatmap(data_list, data_name_list):
    for data, data_name in zip(data_list, data_name_list):
        fig, ax = plt.subplots(1, 1)
        
        #데이터셋의 correlation 계산
        data_cor = data.corr()
        
        #각 feature의 correlation을 heatmap으로 그림
        sns.heatmap(data_cor, annot=True, fmt='.2f')
        ax.set_title(data_name)