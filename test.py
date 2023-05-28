import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

plt.rcParams['font.family'] = 'Malgun Gothic'
pd.set_option("display.max_rows", 25)
# ==============================================
# 1. 데이터셋 로드
# ==============================================

# ----------------------------------------------
# 주요 채소류 일일가격(19960128-20230506) 데이터셋
# ----------------------------------------------
# 배추, 무, 마늘, 양파, 건고추 등 5개 품목
#
# inq_ymd 조회일자
# pdlt_nm 품목명
# whsl_dail_prce 소매일일가격
# rtsl_dail_prce 소매일일가격
# ----------------------------------------------
price_df = pd.read_csv("original_dataset/TB_TAT_DAILY_SOON_PRC_CRS.csv", low_memory=False)

# ----------------------------------------------
# 기상관측(19960128-20230519) 데이터셋
# ----------------------------------------------
# 지점d
# ----------------------------------------------
asos_df1 = pd.read_csv("original_dataset/OBS_ASOS_DD_19960128_20060128.csv", low_memory=False)
asos_df2 = pd.read_csv("original_dataset/OBS_ASOS_DD_20060129_20120129.csv", low_memory=False)
asos_df3 = pd.read_csv("original_dataset/OBS_ASOS_DD_20120130_20220130.csv", low_memory=False)
asos_df4 = pd.read_csv("original_dataset/OBS_ASOS_DD_20220131_20230519.csv", low_memory=False)
total_asos_df = pd.concat([asos_df1, asos_df2, asos_df3, asos_df4])

# ==============================================
# 2. 데이터셋 정리
# ==============================================

# ----------------------------------------------
# 주요 채소류 일일가격(19960128-20230506) 데이터셋
# ----------------------------------------------
# 각 품목에 따라 일일 가격 정리
# column 이름 재설정
# 데이터 타입 변경
# ----------------------------------------------

# 1) 배추 가격 데이터셋
napa_cabbage_df = price_df[price_df['PDLT_NM'] == "배추"]
napa_cabbage_df = napa_cabbage_df[['INQ_YMD', 'PDLT_NM', 'RTSL_DAIL_PRCE']]
napa_cabbage_df.columns = ['일시', '품목', '소매일일가격']
napa_cabbage_df['일시'] = pd.to_datetime(napa_cabbage_df['일시'], format="%Y%m%d")
napa_cabbage_df = napa_cabbage_df.astype({'일시':'str'})
print(napa_cabbage_df.head())

# 2) 무 가격 데이터셋
radish_df = price_df[price_df['PDLT_NM'] == "무"]
radish_df = radish_df[['INQ_YMD', 'PDLT_NM', 'RTSL_DAIL_PRCE']]
radish_df.columns = ['일시', '품목', '소매일일가격']
radish_df['일시'] = pd.to_datetime(radish_df['일시'], format="%Y%m%d")
radish_df = radish_df.astype({'일시':'str'})

# 3) 마늘 가격 데이터셋
garlic_df = price_df[price_df['PDLT_NM'] == "마늘"]
garlic_df = garlic_df[['INQ_YMD', 'PDLT_NM', 'RTSL_DAIL_PRCE']]
garlic_df.columns = ['일시', '품목', '소매일일가격']
garlic_df['일시'] = pd.to_datetime(garlic_df['일시'], format="%Y%m%d")
garlic_df = garlic_df.astype({'일시':'str'})

# 4) 건고추 가격 데이터셋
pepper_df = price_df[price_df['PDLT_NM'] == "건고추"]
pepper_df = pepper_df[['INQ_YMD', 'PDLT_NM', 'RTSL_DAIL_PRCE']]
pepper_df.columns = ['일시', '품목', '소매일일가격']
pepper_df['일시'] = pd.to_datetime(pepper_df['일시'], format="%Y%m%d")
pepper_df = pepper_df.astype({'일시':'str'})

# ----------------------------------------------
# 기상관측(19960128-20230519) 데이터셋
# ----------------------------------------------
# 쓸모없는 attribute 정리
# ----------------------------------------------

total_assos_df = total_asos_df[['일시','평균기온(°C)','최저기온(°C)', '최고기온(°C)', "최소 상대습도(%)", "평균 상대습도(%)", '최대 풍속(m/s)', "평균 풍속(m/s)", '합계 일사량(MJ/m2)', '합계 일조시간(hr)', "평균 지면온도(°C)"]]
grouped = total_assos_df.groupby('일시').mean().reset_index()

print(grouped.head())

# ==============================================
# 3. 데이터셋 합치기
# ==============================================
# 주요 채소류 일일가격 데이터셋 각 품목 + 기상관측 데이터셋
# 일시를 기준으로 합치기
#-----------------------------------------------

# 1) 배추 가격 데이터셋 + 기상관측 데이터셋
napa_cabbage_plus_assos_df = pd.merge(napa_cabbage_df, grouped)
napa_cabbage_plus_assos_df.drop(napa_cabbage_plus_assos_df[napa_cabbage_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
napa_cabbage_plus_assos_df.to_csv("napa_cabbage_plus_assos_df.csv", encoding="utf-8")
print(napa_cabbage_plus_assos_df.head())

# 2) 무 가격 데이터셋 + 기상관측 데이터셋
radish_plus_assos_df = pd.merge(radish_df, grouped)
radish_plus_assos_df.drop(radish_plus_assos_df[radish_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
radish_plus_assos_df.to_csv("radish_plus_assos_df.csv", encoding="utf-8")
print(radish_plus_assos_df.head())

# 3) 마늘 가격 데이터셋 + 기상관측 데이터셋
garlic_plus_assos_df = pd.merge(garlic_df, grouped)
garlic_plus_assos_df.drop(garlic_plus_assos_df[garlic_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
garlic_plus_assos_df.to_csv("garlic_plus_assos_df.csv", encoding="utf-8")
print(garlic_plus_assos_df.head())

# 4) 건고추 가격 데이터셋 + 기상관측 데이터셋
pepper_plus_assos_df = pd.merge(pepper_df, grouped)
pepper_plus_assos_df.drop(pepper_plus_assos_df[pepper_plus_assos_df['소매일일가격'] == 0].index, inplace=True)
pepper_plus_assos_df.to_csv("pepper_plus_assos_df.csv", encoding="utf-8")
print(pepper_plus_assos_df.head())

# ==============================================
# 4. feature들 간의 상관관계 계산
# ==============================================
# 1) 배추
napa_corr = napa_cabbage_plus_assos_df.corr()
sns.heatmap(napa_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 2) 무
radish_corr = radish_plus_assos_df.corr()
sns.heatmap(radish_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 3) 마늘
garlic_corr = garlic_plus_assos_df.corr()
sns.heatmap(garlic_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

# 4) 건고추
pepper_corr = pepper_plus_assos_df.corr()
sns.heatmap(pepper_corr, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()



# ==============================================
# 5. 통계적 분석
# ==============================================

# Calculate mean, median, mode, standard deviation, and variance
# 1) 배추
mean = napa_cabbage_plus_assos_df['소매일일가격'].mean()
median = napa_cabbage_plus_assos_df['소매일일가격'].median()
mode = napa_cabbage_plus_assos_df['소매일일가격'].mode()
std_dev = napa_cabbage_plus_assos_df['소매일일가격'].std()
variance = napa_cabbage_plus_assos_df['소매일일가격'].var()

print("배추 소매일일가격 분석")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# 2) 무
mean = radish_plus_assos_df['소매일일가격'].mean()
median = radish_plus_assos_df['소매일일가격'].median()
mode = radish_plus_assos_df['소매일일가격'].mode()
std_dev = radish_plus_assos_df['소매일일가격'].std()
variance = radish_plus_assos_df['소매일일가격'].var()

print("무 소매일일가격 분석")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# 3) 마늘
mean = garlic_plus_assos_df['소매일일가격'].mean()
median = garlic_plus_assos_df['소매일일가격'].median()
mode = garlic_plus_assos_df['소매일일가격'].mode()
std_dev = garlic_plus_assos_df['소매일일가격'].std()
variance = garlic_plus_assos_df['소매일일가격'].var()

print("마늘 소매일일가격 분석")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# 4) 건고추
mean = pepper_plus_assos_df['소매일일가격'].mean()
median = pepper_plus_assos_df['소매일일가격'].median()
mode = pepper_plus_assos_df['소매일일가격'].mode()
std_dev = pepper_plus_assos_df['소매일일가격'].std()
variance = pepper_plus_assos_df['소매일일가격'].var()

print("건고추 소매일일가격 분석")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# ==============================================
# 6. outlier 탐지
# ==============================================

df_list = [napa_cabbage_plus_assos_df, radish_plus_assos_df, garlic_plus_assos_df, pepper_plus_assos_df]
vegetable_list = ["배추", "무", "마늘", "건고추"]
feature_list = ['평균기온(°C)','최저기온(°C)', '최고기온(°C)', "최소 상대습도(%)", "평균 상대습도(%)", '최대 풍속(m/s)', "평균 풍속(m/s)", '합계 일사량(MJ/m2)', '합계 일조시간(hr)', "평균 지면온도(°C)"]

for df, vegetable in zip(df_list, vegetable_list):
    plt.subplot()
    plt.suptitle(vegetable)
    for idx, feature in enumerate(feature_list):               
        # Select the feature(s) to be used for outlier detection
        features = ['소매일일가격', feature]

        # Create a scaler object to normalize the features
        scaler = MinMaxScaler()

        # Normalize the selected features
        normalized_features = scaler.fit_transform(napa_cabbage_plus_assos_df[features])

        # Create an instance of the LOF outlier detection model
        model = LOF(contamination=0.05)  # Adjust the contamination parameter as needed

        # Fit the model to the normalized features
        model.fit(normalized_features)

        # Predict the outlier scores
        outlier_scores = model.decision_scores_

        # Add the outlier scores to the dataset
        napa_cabbage_plus_assos_df['Outlier Score'] = outlier_scores

        # Print the instances considered as outliers
        outliers = napa_cabbage_plus_assos_df[napa_cabbage_plus_assos_df['Outlier Score'] > model.threshold_]

        plt.subplot(2,5,idx+1)
        plt.title(feature)
        plt.scatter(napa_cabbage_plus_assos_df[feature], napa_cabbage_plus_assos_df['소매일일가격'], c='blue', label='Inliers')
        plt.scatter(outliers[feature], outliers['소매일일가격'], c='red', label='Outliers')
        plt.xlabel(feature)
        plt.ylabel('소매일일가격')
        plt.legend()
    plt.show()
