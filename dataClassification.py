import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# 1. Change the data type to date
# 2. Remove data from 1999
# 3. Keep only November and December data
# 4. Divide the data into high, medium, and low based on price and add a new feature
# 5. Save the completed data as a CSV file

def final_df_classification(df_list, name_list):
    for df, name in zip(df_list, name_list):
        df['일시'] = pd.to_datetime(df['일시'])

        # Remove data from 1999
        df = df[df['일시'].dt.year != 1999]

        # Keep only November and December data
        df = df[(df['일시'].dt.month == 11) | (df['일시'].dt.month == 12)]

        # Divide the data into high, medium, and low based on price and add a new feature
        point1 = df['인플레이션 반영가'].min() + (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min()) / 3
        point2 = df['인플레이션 반영가'].min() + 2 * (df['인플레이션 반영가'].max() - df['인플레이션 반영가'].min()) / 3

        # Initialize the new feature
        df['가격'] = -1

        # Set values less than point1 as 1
        df.loc[df['인플레이션 반영가'] < point1, '가격'] = 1

        # Set values greater than or equal to point1 and less than point2 as 2
        df.loc[(df['인플레이션 반영가'] >= point1) & (df['인플레이션 반영가'] < point2), '가격'] = 2

        # Set values greater than or equal to point2 as 3
        df.loc[df['인플레이션 반영가'] >= point2, '가격'] = 3

        df.to_csv("final_df_classification/" + name + "_df.csv", encoding="utf-8")


def knn_classification(df_list, name_list, k):
    acc_scale_model = {}

    for df, name in zip(df_list, name_list):
        df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', '일시', '품목'], axis=1, inplace=True)
        df = df[['연도', '인플레이션 반영가', '가격']]

        df_train = df[df['연도'] < 2021]  # Changed the order of splitting
        df_test = df[df['연도'] >= 2021]   # Changed the order of splitting
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
