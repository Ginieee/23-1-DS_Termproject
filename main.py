import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from merge_df import extract_datetime, rename_features, merge_dataset
from addDirtyData import setting, addDirtyData
from removeDirtyData import remove_save
from removeOutlier import removeOutliers
from correlation import draw_corr_heatmap, setting2
from exploration import data_exploration, setting_exploration
from inflation import reflect_inflation
from algorithm import find_best_feature_combination, run_multipleRegression, visualizeDistribution, add_previous_feature
from dataClassification import final_df_classification, knn_classification
from kmeans_algorithm import perform_pca, plot_cumulative_variance_ratio, multiple_kmeans_algorithm, calculate_cumulative_variance_ratio, do_multiple_kmeans

plt.rcParams['font.family'] = 'Malgun Gothic'
# ==============================================
# 1. Load Datasets
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
napa_cabbage_plus_assos_df = pd.read_csv("original_dataset/weather/napa_cabbage_plus_assos_df.csv", low_memory=False)
radish_plus_assos_df = pd.read_csv("original_dataset/weather/radish_plus_assos_df.csv", low_memory=False)
galric_plus_assos_df = pd.read_csv("original_dataset/weather/garlic_plus_assos_df.csv", low_memory=False)
pepper_plus_assos_df = pd.read_csv("original_dataset/weather/pepper_plus_assos_df.csv", low_memory=False)

# ----------------------------------------------
# 강어진 데이터셋(소매가격)
# ----------------------------------------------
napa_cabbage_plus_price_df = pd.read_csv("original_dataset/price/df_cabbage.csv", low_memory=False, encoding='cp949')
radish_plus_price_df = pd.read_csv("original_dataset/price/df_radish.csv", low_memory=False, encoding='cp949')
galric_plus_price_df = pd.read_csv("original_dataset/price/df_garlic.csv", low_memory=False, encoding='cp949')
pepper_plus_price_df = pd.read_csv("original_dataset/price/df_pepper.csv", low_memory=False, encoding='cp949')

# ----------------------------------------------
# 장원준 데이터셋(소매가격 + 수입,수출)
# ----------------------------------------------
napa_cabbage_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_cabbage.xlsx")
radish_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_radish.xlsx")
galric_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_garlic.xlsx")
pepper_plus_income_export_df = pd.read_excel("original_dataset/income/income_export_pepper.xlsx")

assos_df_list = [napa_cabbage_plus_assos_df, radish_plus_assos_df, galric_plus_assos_df, pepper_plus_assos_df]
income_export_df_list = [napa_cabbage_plus_income_export_df, radish_plus_income_export_df, galric_plus_income_export_df, pepper_plus_income_export_df]

# ==============================================
# 2. Merge Datasets
# ==============================================
print('Merge Datasets ------------------------------------------------------------------------------------------------------------------------------------------------------------')
extract_datetime(assos_df_list)
rename_features(income_export_df_list)
merge_dataset(assos_df_list, income_export_df_list)

# ==============================================
# 3. Add Dirty Data
# ==============================================
print('Add Dirty Data ------------------------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("merged_data/garlic_merged_list.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("merged_data/napa_cabbage_merged_list.csv", low_memory=False)
radish_df = pd.read_csv("merged_data/radish_merged_list.csv", low_memory=False)
pepper_df = pd.read_csv("merged_data/pepper_merged_list.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]

setting(df_list)
addDirtyData(df_list, df_name_list)

# ==============================================
# 4. Data Exploration
# ==============================================
print('Data Exploration ----------------------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

setting_exploration(df_list)
data_exploration(df_list, name_list)

# ==============================================
# 5. Remove Dirty Data
# ==============================================
print('Remove Dirty Data ---------------------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
item_list = ['마늘', '배추', '무', '건고추']
file_path = "remove_dirtyData/"

remove_save(df_list, df_name_list, item_list, file_path)

# ==============================================
# 6. Remove Outliers
# ==============================================
print('Remove Outliers -----------------------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("remove_dirtyData/garlic_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("remove_dirtyData/napa_cabbage_df.csv", low_memory=False)
radish_df = pd.read_csv("remove_dirtyData/radish_df.csv", low_memory=False)
pepper_df = pd.read_csv("remove_dirtyData/pepper_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
columns = ["평균기온(°C)","최저기온(°C)","최고기온(°C)","최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)","수출(kg)","수출(달러)","수입(kg)","수입(달러)"]
file_path = "remove_outlier/"

plt.rcParams['font.family'] = 'Malgun Gothic'

removeOutliers(df_list, df_name_list, columns, file_path)

# ==============================================
# 7. Correlation amongst features
# ==============================================
print('Correlation among features ------------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("remove_outlier/garlic_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("remove_outlier/napa_cabbage_df.csv", low_memory=False)
radish_df = pd.read_csv("remove_outlier/radish_df.csv", low_memory=False)
pepper_df = pd.read_csv("remove_outlier/pepper_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

setting2(df_list)
draw_corr_heatmap(df_list, name_list)
plt.show()

# ==============================================
# 8. Reflect inflation on data
# ==============================================
# Taking Inflation into DataFrame
print('Read Inflation Data ----------------------------------------------------------------------------------------------------------------------------------')
inflation_df = pd.read_excel("original_dataset/consumer_price_index.xlsx")

print('Data Exploration of Inflation Data ----------------------------------------------------------------------------------------------------------------------')
data_exploration([inflation_df], ["Inflation DataFrame"])

print('Reflect inflation on target -------------------------------------------------------------------------------------------------------------')
reflect_inflation(df_list, inflation_df)

# ==============================================
# 9. Correlation amongst features with inflation
# ==============================================
print('Correlation among features with inflation ----------------------------------------------------------------------------------------------------------------')
draw_corr_heatmap(df_list, name_list)
plt.show()

# ==============================================
# 10.
# ==============================================

#add_previous_feature(df_list, item_list)

garlic_df = pd.read_csv("add_previous_feature/마늘_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_previous_feature/배추_df.csv", low_memory=False)
radish_df = pd.read_csv("add_previous_feature/무_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_previous_feature/건고추_df.csv", low_memory=False)


print('Previous price--------------------------------------------------------------------------------------------------------------------------------------------')
garlic_df = pd.read_csv("add_price/마늘_price_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_price/배추_price_df.csv", low_memory=False)
radish_df = pd.read_csv("add_price/무_price_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_price/건고추_price_df.csv", low_memory=False)
df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
name_list = ["garlic", "napa_cabbage", "radish", "pepper"]
data_exploration(df_list, name_list)

# ==============================================
# 12. KMeans Clustering
# ==============================================
print('Multiple KMeans--------------------------------------------------------------------------------------------------------------------------------------------')
do_multiple_kmeans(df_list, name_list)
plt.show()