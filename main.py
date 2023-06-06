import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from merge_df import extract_datetime, rename_features, merge_dataset
from addDirtyData import setting, addDirtyData
<<<<<<< Updated upstream
from removeDirtyData import remove_save
from removeOutlier import removeOutliers
from correlation import draw_corr_heatmap, setting2
from exploration import data_exploration, setting_exploration
from inflation import reflact_inflation
from algorithm import find_best_feature_combination, run_multipleRegression, visualizeDistribution, add_previous_feature
=======
from removeDirtyData import remove_save, compare_dirty_data
from removeOutlier import removeOutliers, compare_outliers
from correlation import draw_corr_heatmap, setting2, drop_non_numeric_Features
from exploration import data_exploration, setting_exploration
from inflation import reflect_inflation
from algorithm import find_best_feature_combination
from dataClassification import final_df_classification, knn_classification
from kmeans_algorithm import perform_pca, plot_cumulative_variance_ratio, multiple_kmeans_algorithm, calculate_cumulative_variance_ratio, do_multiple_kmeans
from add_previous_value import add_previous_feature, add_previous_price_feature, add_previous_save
from run_linear_regression import run_multiple_linear_regression,run_linear_regression
from run_polynomial_regression import run_polynomial_regression
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler, Normalizer
>>>>>>> Stashed changes

plt.rcParams['font.family'] = 'Malgun Gothic'
# # ==============================================
# # 1. Load Datasets
# # ==============================================

# # ----------------------------------------------
# # 이지해 데이터셋(소매가격 + 기상관측)
# # ----------------------------------------------
# # 배추, 무, 마늘, 건고추
# #
# # Feature)
# # 일시
# # 품목
# # 소매일일가격
# # 평균기온(°C)
# # 최저기온(°C)
# # 최고기온(°C)
# # 최소 상대습도(%)
# # 평균 상대습도(%)
# # 최대 풍속(m/s)
# # 평균 풍속(m/s)
# # 합계 일사량(MJ/m2)
# # 합계 일조시간(hr)
# # 평균 지면온도(°C)
# # ----------------------------------------------
# napa_cabbage_plus_assos_df = pd.read_csv("original_dataset/weather/napa_cabbage_plus_assos_df.csv", low_memory=False)
# radish_plus_assos_df = pd.read_csv("original_dataset/weather/radish_plus_assos_df.csv", low_memory=False)
# galric_plus_assos_df = pd.read_csv("original_dataset/weather/garlic_plus_assos_df.csv", low_memory=False)
# pepper_plus_assos_df = pd.read_csv("original_dataset/weather/pepper_plus_assos_df.csv", low_memory=False)

# # ----------------------------------------------
# # 강어진 데이터셋(소매가격)
# # ----------------------------------------------
# napa_cabbage_plus_price_df = pd.read_csv("original_dataset/price/df_cabbage.csv", low_memory=False, encoding='cp949')
# radish_plus_price_df = pd.read_csv("original_dataset/price/df_radish.csv", low_memory=False, encoding='cp949')
# galric_plus_price_df = pd.read_csv("original_dataset/price/df_garlic.csv", low_memory=False, encoding='cp949')
# pepper_plus_price_df = pd.read_csv("original_dataset/price/df_pepper.csv", low_memory=False, encoding='cp949')

# # ----------------------------------------------
# # 장원준 데이터셋(소매가격 + 수입,수출)
# # ----------------------------------------------
# napa_cabbage_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_cabbage.xlsx")
# radish_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_radish.xlsx")
# galric_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_garlic.xlsx")
# pepper_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_pepper.xlsx")

# assos_df_list = [napa_cabbage_plus_assos_df, radish_plus_assos_df, galric_plus_assos_df, pepper_plus_assos_df]
# income_export_df_list = [napa_cabbage_plus_income_export_df, radish_plus_income_export_df, galric_plus_income_export_df, pepper_plus_income_export_df]

# # ==============================================
# # 2. Merge Datasets
# # ==============================================
# print('Merge Datasets ------------------------------------------------------------------------------------------------------------------------------------------------------------')
# extract_datetime(assos_df_list)
# rename_features(income_export_df_list)
# merge_dataset(assos_df_list, income_export_df_list)


# # ==============================================
# # 3. Add Dirty Data
# # ==============================================
# print('Add Dirty Data ------------------------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("merged_data/garlic_merged_list.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("merged_data/napa_cabbage_merged_list.csv", low_memory=False)
# radish_df = pd.read_csv("merged_data/radish_merged_list.csv", low_memory=False)
# pepper_df = pd.read_csv("merged_data/pepper_merged_list.csv", low_memory=False)

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]

# setting(df_list)
# addDirtyData(df_list, df_name_list)

# # ==============================================
# # 4. Data Exploration
# # ==============================================
# print('Data Exploration ----------------------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

# setting_exploration(df_list)
# data_exploration(df_list, name_list)

# # ==============================================
# # 5. Remove Dirty Data
# # ==============================================
# print('Remove Dirty Data ---------------------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
item_list = ['마늘', '배추', '무', '건고추']
# file_path = "remove_dirtyData/"

# after_remove_dirty_data_list = remove_save(df_list.copy(), df_name_list, item_list, file_path)
# # compare_dirty_data(df_list, after_remove_dirty_data_list, item_list)
# df_list = after_remove_dirty_data_list

# # ==============================================
# # 6. Reflect inflation on data
# # ==============================================
# # Taking Inflation into DataFrame
# print('Read Inflation Data ----------------------------------------------------------------------------------------------------------------------------------')
# inflation_df = pd.read_excel("original_dataset/consumer_price_index.xlsx")

# print('Data Exploration of Inflation Data ----------------------------------------------------------------------------------------------------------------------')
# data_exploration([inflation_df], ["Inflation DataFrame"])

# print('Reflect inflation on target -------------------------------------------------------------------------------------------------------------')
# reflect_inflation(df_list, inflation_df)

# ==============================================
# 7. add previous feature
# ==============================================
# print('Add Previous Feature ---------------------------------------------------------------------------------------------------------------------------------------------------------')

# garlic_df = pd.read_csv("reflect_inflation/inflation_garlic_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("reflect_inflation/inflation_cabbage_df.csv", low_memory=False)
# radish_df = pd.read_csv("reflect_inflation/inflation_radish_df.csv", low_memory=False)
# pepper_df = pd.read_csv("reflect_inflation/inflation_pepper_df.csv", low_memory=False)

# file_path = "add_previous_feature/"

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]

# previous_df_list = add_previous_feature(df_list, item_list)

# garlic_df = pd.read_csv("add_previous_feature/마늘_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_previous_feature/배추_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_previous_feature/무_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_previous_feature/건고추_df.csv", low_memory=False)

# previous_df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# # add_previous_save(previous_df_list, item_list, file_path, "_df")

# previous_df_list = add_previous_price_feature(previous_df_list)

# add_previous_save(previous_df_list, item_list, file_path, "_price_df")
# df_list = previous_df_list


# ==============================================
# 7. Remove Outliers
# ==============================================
print('Remove Outliers -----------------------------------------------------------------------------------------------------------------------------------------------------------')
#  올해 가격 예측을 위한 data 따로 빼 놓기

garlic_df = pd.read_csv("add_previous_feature/마늘_price_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_previous_feature/베추_price_df.csv", low_memory=False)
radish_df = pd.read_csv("add_previous_feature/무_price_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_previous_feature/고추_price_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]
df_for_predict_list = []

for i, (df, item) in enumerate(zip(df_list, name_list)):
    if item == "Pepper":
        df_for_predict = df[df['연도' == 2012]]
        df_for_predict = df_for_predict[df_for_predict['월' == 12]]
        
        df_for_predict_list.append(df_for_predict)
        
        df = df[not(df['연도' == 2012] and df['월' == 12])]
        
        df_list[i] = df
        print(df)
    else: 
        df_for_predict = df[df['연도' == 2023]]
        df_for_predict = df_for_predict[df_for_predict['월' == 4]]
        
        df_for_predict_list.append(df_for_predict)
        
        df = df[not(df['연도' == 2023] and df['월' == 4])]
        
        df_list[i] = df
        print(df)

df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
columns = ["평균기온(°C)","최저기온(°C)","최고기온(°C)","최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)","수출(kg)","수출(달러)","수입(kg)","수입(달러)"]
file_path = "remove_outlier/"

data_list_after = removeOutliers(df_list.copy(), df_name_list, columns, file_path)
df_list = data_list_after
# print(garlic_df['최대 풍속(m/s)'].describe())
# print(data_list_after[0]['최대 풍속(m/s)'].describe())
compare_outliers(df_list.copy(), data_list_after, ['수입(kg)', '최대 풍속(m/s)', '평균 상대습도(%)'])

# # # ==============================================
# # # 8. Correlation amongst features
# # # ==============================================
# # print('Correlation among features ------------------------------------------------------------------------------------------------------------------------------------------------')
# # garlic_df = pd.read_csv("remove_outlier/garlic_df.csv", low_memory=False)
# # napa_cabbage_df = pd.read_csv("remove_outlier/napa_cabbage_df.csv", low_memory=False)
# # radish_df = pd.read_csv("remove_outlier/radish_df.csv", low_memory=False)
# # pepper_df = pd.read_csv("remove_outlier/pepper_df.csv", low_memory=False)

# # df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# # name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

# # df_list = drop_non_numeric_Features(df_list)
# # setting2(df_list)
# # draw_corr_heatmap(df_list, name_list, "소매일일가격")
# # plt.show()

<<<<<<< Updated upstream
setting2(df_list)
draw_corr_heatmap(df_list, name_list)
# plt.show()

# ==============================================
# 8. Reflect inflation on data
# ==============================================
# Taking Inflation into DataFrame
print('Read Inflation Data ----------------------------------------------------------------------------------------------------------------------------------------------------------')
inflation_df = pd.read_excel("original_dataset/consumer_price_index.xlsx")

print('Data Exploration of Inflation Data--------------------------------------------------------------------------------------------------------------------------------------------')
data_exploration([inflation_df], ["Inflation DataFrame"])

print('Reflect inflation on target--------------------------------------------------------------------------------------------------------------------------------------------')
reflact_inflation(df_list, inflation_df)

# ==============================================
# 9. Correlation amongst features with inflation
# ==============================================
# print('Correlation among features with inflation------------------------------------------------------------------------------------------------------------------------------------------------')
# draw_corr_heatmap(df_list, name_list)
# plt.show()

# ==============================================
# 10. 
# ==============================================
for df in df_list:
    print(df)
    print("drop")
    df.dropna(subset=['인플레이션 반영가'], axis = 0, inplace=True)
    print(df)
    print("--------------")

add_previous_feature(df_list, item_list)
=======
# # # ==============================================
# # # 10. Correlation amongst features with inflation
# # # ==============================================
# # print('Correlation among features with inflation ----------------------------------------------------------------------------------------------------------------')
# # df_list = drop_non_numeric_Features(df_list)
# # draw_corr_heatmap(df_list, name_list, "인플레이션 반영가")
# # plt.show()

# # ===============================================
# # 11. Drop unusable features
# # ===============================================

# # 올해 가격 예측을 위한 data 따로 빼 놓기
# df_for_predict_list = []

# for i, (df, item) in enumerate(zip(df_list, name_list)):
#     if item == "Pepper":
#         df_for_predict = df[df['연도' == 2012]]
#         df_for_predict = df_for_predict[df_for_predict['월' == 12]]
        
#         df_for_predict_list.append(df_for_predict)
        
#         df = df[not(df['연도' == 2012] and df['월' == 12])]
        
#         df_list[i] = df
#         print(df)
#     else: 
#         df_for_predict = df[df['연도' == 2023]]
#         df_for_predict = df_for_predict[df_for_predict['월' == 4]]
        
#         df_for_predict_list.append(df_for_predict)
        
#         df = df[not(df['연도' == 2023] and df['월' == 4])]
        
#         df_list[i] = df
#         print(df)
#     assert(0)

# for i, df in enumerate(df_list):
    
#     df.replace(-1.0, np.nan,inplace=True)

#     df.fillna(method='ffill', inplace=True)

#     df = df[df['연도'] != 1999]
#     df.drop('소매일일가격', axis=1, inplace=True)
#     df.drop('일시', axis=1, inplace=True)
#     # df.drop('연도', axis=1, inplace=True)
#     df.drop('품목', axis=1, inplace=True)
#     df.drop('소비자물가총지수', axis=1, inplace=True)
#     df.drop('Unnamed: 0.1', axis=1, inplace=True)
#     df.drop('Unnamed: 0', axis=1, inplace=True)
#     drop_list = ['직전 1달 수출(달러)','직전 1달 수입(달러)','직전 1달 평균기온(°C)','직전 1달 평균 상대습도(%)','직전 1달 합계 일사량(MJ/m2)','직전 1달 평균 풍속(m/s)','직전 1달 최대 풍속(m/s)','직전 1달 최고기온(°C)','직전 1달 최저기온(°C)','직전 1달 평균 지면온도(°C)',
#                  '평균기온(°C)','최저기온(°C)','최고기온(°C)','최소 상대습도(%)','평균 상대습도(%)','최대 풍속(m/s)','평균 풍속(m/s)','합계 일사량(MJ/m2)','합계 일조시간(hr)','평균 지면온도(°C)','수출(kg)','수출(달러)','수입(kg)','수입(달러)']
#     df.drop(drop_list, axis=1, inplace=True)
#     df_list[i] = df
#     print(df.shape)

# data_exploration(df_list, name_list)

# # ===========================================
# # 12. Scaling
# # ===========================================
# target = "인플레이션 반영가"
# selected_scaler = StandardScaler

# for i, (df, df_for_predict) in enumerate(zip(df_list, df_for_predict_list)):
#     scaler = selected_scaler()
#     df_x = df.drop(target, axis=1)
#     df_y = df[target]
#     df_x_scaled = scaler.fit_transform(df_x.iloc[:, 1:])
    
#     df_for_predict_x = df.drop(target, axis=1)
#     df_for_predict_y = df[target]
#     df_for_predict_x_scaled = scaler.transform(df_for_predict_x[:, 1:])

#     df_x_scaled[target] = df_y
#     df_list[i] = df_x_scaled

#     df_for_predict_x_scaled[target] = df_for_predict_y
#     df_for_predict_list[i] = df_for_predict_x_scaled

#     print(df_x_scaled)
#     print(df_for_predict_x_scaled)
#     print("====================================")

# # ==============================================
# # 13. Select Kbest
# # ==============================================

# # target(Price)와 가장 correlated 된 features 를 k개 고르기.

# from sklearn.feature_selection import f_regression, SelectKBest

# selected_feature_list = []

# for df in df_list:
#     x = df.drop('인플레이션 반영가', axis=1)
#     y = df['인플레이션 반영가']
#     ## selctor 정의하기.
#     selector = SelectKBest(score_func=f_regression, k=40)
#     ## 학습데이터에 fit_transform 
#     X_selected = selector.fit_transform(x,y)
#     ## 테스트 데이터는 transform
#     # X_test_selected = selector.transform(X_test)
#     print(X_selected.shape)

#     all_names = x.columns
#     ## selector.get_support()
#     selected_mask = selector.get_support()
#     ## 선택된 특성(변수)들
#     selected_names = all_names[selected_mask]
#     ## 선택되지 않은 특성(변수)들
#     unselected_names = all_names[~selected_mask]
#     print('Selected names: ', selected_names)
#     print('Unselected names: ', unselected_names)
#     print("-----------------------------------------------------\n")
#     selected_feature_list.append(selected_names)



# # ==============================================
# # 14. KMeans Clustering
# # ==============================================
# # print('Multiple KMeans--------------------------------------------------------------------------------------------------------------------------------------------')
# # do_multiple_kmeans(df_list, name_list)

# # ==============================================
# # 15. Regression model - 1) multiple linear regression
# # ==============================================
# print('Multiple Linear Regression--------------------------------------------------------------------------------------------------------------------------------------------')

# model_list = run_multiple_linear_regression(df_list, name_list, selected_feature_list)

# # ==============================================
# # 16. Regression model - 2) multiple linear regression
# # ==============================================
# # print('Polynomial Regression--------------------------------------------------------------------------------------------------------------------------------------------')
# # run_polynomial_regression(df_list, name_list, "인플레이션 반영가", degree=2)
# # run_linear_regression(df_list, name_list)

# # ==============================================
# # 17. 예측) 모델 사용해보기
# # ==============================================

# print('Previous price--------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("add_price/마늘_price_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_price/배추_price_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_price/무_price_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_price/건고추_price_df.csv", low_memory=False)
# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# name_list = ["garlic", "napa_cabbage", "radish", "pepper"]

# for df, model in zip(df_list, model_list):
#     y = df['인플레이션 반영가']
#     x = df.drop('인플레이션 반영가', axis=1)

#     predict = model.predict(x)
#     print("예측한 값: ", predict)
#     print("실제 값: ", y)
>>>>>>> Stashed changes
