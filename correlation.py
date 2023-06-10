import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def setting2(data_list):
    for data in data_list:
        data.drop("Unnamed: 0", axis=1, inplace=True)
        data.drop("Unnamed: 0.1", axis=1, inplace=True)
        data.drop("Unnamed: 0.1.1", axis=1, inplace=True)


def draw_corr_heatmap(data_list, data_name_list, target):
    for data, data_name in zip(data_list, data_name_list):
        scale_list = ["standard", "minmax", "robust"] # do 3 scaling
        for scale in scale_list:
            df = scale_dataframe(data, scale, target)

            fig, ax = plt.subplots(1, 1)

            # 데이터셋의 correlation 계산
            data_cor = df.corr()

            # 각 feature의 correlation을 heatmap으로 그림
            sns.heatmap(data_cor, annot=True, fmt='.2f')
            ax.set_title("Correlation with {} scaling : {}".format(scale, data_name))


def drop_non_numeric_Features(data_list):
    result = []
    for data in data_list:
        numeric_columns = data.select_dtypes(include='number').columns # save only numeric values
        data = data[numeric_columns]
        result.append(data)

    return result


# to scaling dataframe
def scale_dataframe(df, scaler, target):
    if scaler == "standard":
        scale = StandardScaler()
    elif scaler == "minmax":
        scale = MinMaxScaler()
    elif scaler == "robust":
        scale = RobustScaler()
    else:
        return "Fail to Scale"

    data = df.drop([target], axis=1).reset_index(drop=True)
    target_column = df[target].reset_index(drop=True)

    scaled_data = scale.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    scaled_df[target] = target_column

    return scaled_df
