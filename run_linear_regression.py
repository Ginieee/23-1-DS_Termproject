import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from algorithm import find_best_feature_combination, multipleRegression, visualizeDistribution, linearRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Malgun Gothic'

def drop_feature_start_with(df, start_with, except_list):
    
    for index, value in df.iteritems(): 

        if start_with in index and index not in except_list:
            df.drop(index, axis=1, inplace = True)

    return df

def load_dataset():
    garlic_df = pd.read_csv("add_previous_feature/마늘_price_df.csv", low_memory=False)
    napa_cabbage_df = pd.read_csv("add_previous_feature/배추_price_df.csv", low_memory=False)
    radish_df = pd.read_csv("add_previous_feature/무_price_df.csv", low_memory=False)
    pepper_df = pd.read_csv("add_previous_feature/건고추_price_df.csv", low_memory=False)

    df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
    item_list = ['마늘', '배추', '무', '건고추']

    garlic_target_df = pd.DataFrame([])
    napa_cabbage_target_df = pd.DataFrame([])
    radish_target_df= pd.DataFrame([])
    pepper_target_df = pd.DataFrame([])

    target_df_list = [garlic_target_df, napa_cabbage_target_df, radish_target_df, pepper_target_df]

    return df_list, target_df_list, item_list

def drop_unusable_feature(df_list, item_list):
    
    garlic_target_df = pd.DataFrame([])
    napa_cabbage_target_df = pd.DataFrame([])
    radish_target_df= pd.DataFrame([])
    pepper_target_df = pd.DataFrame([])

    target_df_list = [garlic_target_df, napa_cabbage_target_df, radish_target_df, pepper_target_df]

    for i, (df, target_df, item) in enumerate(zip(df_list, target_df_list, item_list)):
        start_with = "직전 "
        except_list = ['직전 3달 인플레이션 반영가', "직전 4달 인플레이션 반영가", "직전 5달 인플레이션 반영가","직전 6달 인플레이션 반영가"]
        if item == "건고추":
            for n in range(5, 7):
                df = drop_feature_start_with(df, start_with + str(n) + "달", except_list)
        elif item == "배추":
            for n in range(4, 7):
                df = drop_feature_start_with(df, start_with + str(n) + "달", except_list)
        elif item == "무":
            for n in range(4, 7):
                df = drop_feature_start_with(df, start_with + str(n) + "달", except_list)

        drop_feature_start_with(df, start_with + "1달", except_list)

        drop_list = ["Unnamed: 0","일시","품목","소매일일가격","평균기온(°C)","최저기온(°C)","최고기온(°C)","최소 상대습도(%)",
                    "평균 상대습도(%)","최대 풍속(m/s)","평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)", 
                    "소비자물가총지수"]
        
        df.drop(drop_list, axis=1, inplace=True)

        df = df[df['연도']!=1999]

        df.drop('연도', axis=1, inplace=True)

        target_df = df['인플레이션 반영가']
        df.drop('인플레이션 반영가', axis=1, inplace=True)

        df_list[i] = df
        target_df_list[i] = target_df
        print(df)
        print('target')
        print(target_df)

    return df_list, target_df_list

# Apply PCA to each dataframe
def apply_PCA(df_list, target, n_components):
    for i, df in enumerate(df_list):
        
        scaler = StandardScaler()
        x = df.drop(target, axis=1)
        y = df[target]
        x_scled = scaler.fit_transform(x.iloc[:, 1:])

        # Perform PCA
        pca = PCA(n_components=n_components)  # Specify the desired number of components
        transformed_data = pca.fit_transform(x_scled)

        ratio = pca.explained_variance_
        print('eigenvalue:', pca.explained_variance_)
        print('proportion of variance explained:', pca.explained_variance_ratio_)

        PC_values = np.arange(pca.n_components_) + 1
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # Plot the explained variance ratio
        plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2, label='Individual')
        plt.plot(PC_values, cumulative_variance_ratio, 'bo-', linewidth=2, label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.legend()
        plt.show()

        # # Create a new dataframe with the transformed data
        columns = []
        for idx in range(n_components):
            columns.append("PC"+str(idx+1))
        transformed_df = pd.DataFrame(transformed_data, columns=columns)

        transformed_df[target] = y.reset_index(drop=True)
        
        # # Update the dataframe in the df_list
        df_list[i] = transformed_df

    return df_list


def run_multiple_linear_regression(df_list, item_list):
    model_list = []
    target = '인플레이션 반영가'

    df_list_no_pca = df_list.copy()
        
    # PCA 적용 후 model 학습
    df_list_tmp = apply_PCA(df_list.copy(), target, 5)
    
    # PCA 적용 한 후 모델 학습
    for df, item in zip(df_list_tmp, item_list):
        coefficients, model = multipleRegression(df.copy(), item, train_size=0.8)
        coefficient_mapping = dict(zip(df.columns, coefficients))

        # 내림차순으로 sorting
        sorted_mapping = dict(sorted(coefficient_mapping.items(), key=lambda x: x[1], reverse=True))
        print("Model Coefficient(Apply PCA)---------------------------------------")
        print(sorted_mapping)

        print("-------------------------------------------------------------------")

    for df, item in zip(df_list, item_list):
        coefficients, model = multipleRegression(df.copy(), item, train_size=0.8)
        coefficient_mapping = dict(zip(df.columns, coefficients))
        
        # 내림차순으로 sorting
        sorted_mapping = dict(sorted(coefficient_mapping.items(), key=lambda x: x[1], reverse=True))
        print("Model Coefficient(Not Apply PCA)---------------------------------------")
        print(sorted_mapping)

        print("-------------------------------------------------------------------")

        model_list.append(model)

    return model_list


def run_multiple_linear_regression(df_list, item_list, selected_feature):
    model_list = []
    target = '인플레이션 반영가'
    
    df_list_no_pca = df_list.copy()
        
    # PCA 적용 후 model 학습
    df_list_tmp = apply_PCA(df_list.copy(), target, 5)
    
    # SelectKBest로 선택된 특성들로 학습
    for i, df in enumerate(df_list):
        
        tmp = df[target]
        tmp_df = df[selected_feature + [target]]  # 선택된 특성들과 '인플레이션 반영가' 열을 포함

        tmp = df[target]
        tmp_df = df

        tmp_df[target] = tmp
        df_list_no_pca[i] = tmp_df
    
    # PCA 적용 한 후 모델 학습
    for df, item in zip(df_list_tmp, item_list):
        coefficients, model = multipleRegression(df.copy(), item, train_size=0.8)
        coefficient_mapping = dict(zip(df.columns, coefficients))

        # 내림차순으로 sorting
        sorted_mapping = dict(sorted(coefficient_mapping.items(), key=lambda x: x[1], reverse=True))
        print("Model Coefficient(Apply PCA)---------------------------------------")
        print(sorted_mapping)

        print("-------------------------------------------------------------------")

    for df, item in zip(df_list, item_list):
        coefficients, model = multipleRegression(df.copy(), item, train_size=0.8)
        coefficient_mapping = dict(zip(df.columns, coefficients))
        
        # 내림차순으로 sorting
        sorted_mapping = dict(sorted(coefficient_mapping.items(), key=lambda x: x[1], reverse=True))
        print("Model Coefficient(Not Apply PCA)---------------------------------------")
        print(sorted_mapping)

        print("-------------------------------------------------------------------")

        model_list.append(model)

    return model_list

def run_linear_regression(df_list, item_list):
    df_list, target_df_list = drop_unusable_feature(df_list, item_list)
    for df, target_df, item in zip(df_list, target_df_list, item_list):
        for col_name, col in df.iteritems():
            linearRegression(df[col_name], target_df, item, 0.8, col_name)
    