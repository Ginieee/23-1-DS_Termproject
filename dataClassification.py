import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# 1. date타입 변경
# 2. 1999년 데이터 버림
# 3. 11~12월 데이터만 남김
# 4. 가격별로 상, 중, 하 나눠서 feature 추가
# 5. 완료한 데이터 csv파일로 넘김

def final_df_classification(df_list, name_list):
    for df, name in zip(df_list, name_list):
        df['일시'] = pd.to_datetime(df['일시'])

        # 1999년 데이터 버림
        df = df[df['일시'].dt.year != 1999]

        # 11~12월 데이터만 남김
        df = df[(df['일시'].dt.month == 11) | (df['일시'].dt.month == 12)]

        # 가격별로 상, 중, 하 나눠서 feature 추가
        point1 = df['인플레이션 반영가'].min() + (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min()) / 3
        point2 = df['인플레이션 반영가'].min() + 2 * (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min()) / 3

        # 새로운 feature 초기화
        df['가격'] = -1

        # for idx in range(df.length)
        # point1보다 작은 값은 1으로 설정
        df.loc[df['인플레이션 반영가'] < point1, '가격'] = 1

        # point1보다 크고 point2보다 작은 값은 2로 설정
        df.loc[(df['인플레이션 반영가'] >= point1) & (df['인플레이션 반영가'] < point2), '가격'] = 2

        # point2보다 큰 값은 3으로 설정
        df.loc[df['인플레이션 반영가'] >= point2, '가격'] = 3

        df.to_csv("final_df_classification/" + name + "_df.csv", encoding="utf-8")


def knn_classification(df_list, name_list, k):
    acc_scale_model = {}

    for df, name in zip(df_list, name_list):
        df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', '일시', '품목'], axis=1, inplace=True)
        df = df[['연도', '인플레이션 반영가', '가격']]

        df_train = df[df['연도'] < 2021]  # 분할 순서 변경
        df_test = df[df['연도'] >= 2021]   # 분할 순서 변경
        print('--------{}--------'.format(name))
        print(len(df_train))
        print(len(df_test))

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        train_x = df_train.drop(['가격'], axis=1)
        train_y = df_train['가격']

        test_x = df_test.drop(['가격'], axis=1)
        test_y = df_test['가격']

        scaler = StandardScaler()
        scaled_train_x = scaler.fit_transform(train_x)
        scaled_test_x = scaler.transform(test_x)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(scaled_train_x, train_y)
        predicted_class = knn.predict(scaled_test_x)
        # predict_df = pd.DataFrame({'real' : train_y, 'predict': predicted_class})
        #
        cm = confusion_matrix(test_y, predicted_class)
        print('--------{}--------'.format(name))
        print(cm)

        acc_train = knn.score(scaled_train_x, train_y)
        acc_test = knn.score(scaled_test_x, test_y)

        acc_scale_model[name] = [acc_train, acc_test]

    return acc_scale_model

