from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from run_linear_regression import drop_unusable_feature
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

def apply_PCA_values(df_list, target_list):
    for i, (df, target_df) in enumerate(zip(df_list, target_list)):
        
        scaler = StandardScaler()
        Y = scaler.fit_transform(df.iloc[:, 1:])

        # Perform PCA
        pca = PCA(n_components=6)  # Specify the desired number of components
        transformed_data = pca.fit_transform(Y)

        # # Create a new dataframe with the transformed data
        transformed_df = pd.DataFrame(transformed_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
        
        transformed_df['가격'] = target_df.reset_index(drop=True)
        
        # # Update the dataframe in the df_list
        df_list[i] = transformed_df

    return df_list

def bagging_multiple_regression(df_list, name_list):
    #데이터 전처리
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    #PCA 적용
    df_list = apply_PCA_values(df_list, target_list)
    
    #target, pca로 나눔
    for i in range(len(df_list)):
        target_list[i] = df_list[i]['가격']
        df_list[i] = df_list[i].drop('가격', axis=1)
    
    for df, target, name in zip(df_list, target_list, name_list):
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

        # 기본 분류기 생성
        base_classifier = LinearRegression()

        # Bagging 분류기 생성
        bagging_regression = BaggingRegressor(base_classifier, n_estimators=10, random_state=42)

        # Bagging 분류기 학습
        bagging_regression.fit(X_train, y_train)

        # Bagging 분류기 예측
        y_pred = bagging_regression.predict(X_test)

        # 정확도 평가
        r2 = r2_score(y_test, y_pred)
        print(f"{name} multiple r2 score:", r2)

def bagging_polynomial_regression(df_list, name_list, degree):
    x_poly_list = []
    
    #데이터 전처리
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    
    #polynomial feature 생성
    poly_features = PolynomialFeatures(degree=degree)
    
    for i in range(len(df_list)):
        x_poly_list.append(poly_features.fit_transform(df_list[i]))
    
    for x_poly, target, name in zip(x_poly_list, target_list, name_list):

        # 기본 분류기 생성
        base_classifier = LinearRegression()

        # Bagging 분류기 생성
        bagging_regression = BaggingRegressor(base_classifier, n_estimators=10, random_state=42)

        # Bagging 분류기 학습
        bagging_regression.fit(x_poly, target)

        # Bagging 분류기 예측
        y_pred = bagging_regression.predict(x_poly)

        # 정확도 평가
        r2 = r2_score(x_poly, y_pred)
        print(f"{name} polynomial r2 score:", r2)

#사이킷런 래퍼 XGBoost 모듈
def XGB_multiple_regression(df_list, name_list):
    
    #데이터 전처리
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    #PCA 적용
    df_list = apply_PCA_values(df_list, target_list)
    
    #target, pca로 나눔
    for i in range(len(df_list)):
        target_list[i] = df_list[i]['가격']
        df_list[i] = df_list[i].drop('가격', axis=1)
    
    for df, target, name in zip(df_list, target_list, name_list):
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
        #테스트와 검증 데이터로 나눔
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=12)
        
        #객체 생성
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
        xgb_model = xgb_model.fit(X_train, y_train, early_stopping_rounds=100,
                              eval_metric='logloss', eval_set=[(X_val, y_val)])
        
        #예측 정확도
        score = xgb_model.score(X_val, y_val)
        print(f"{name} score:", score)

#사이킷런 래퍼 XGBoost 모듈
def XGB_polynomial_regression(df_list, name_list, degree):
    x_poly_list = []
    
    #데이터 전처리
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    
    #polynomial feature 생성
    poly_features = PolynomialFeatures(degree=degree)
    
    for i in range(len(df_list)):
        x_poly_list.append(poly_features.fit_transform(df_list[i]))
    
    for x_poly, target, name in zip(x_poly_list, target_list, name_list):
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(x_poly, target, test_size=0.2, random_state=42)
        #테스트와 검증 데이터로 나눔
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=12)
        
        #객체 생성
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
        xgb_model = xgb_model.fit(X_train, y_train, early_stopping_rounds=100,
                              eval_metric='logloss', eval_set=[(X_val, y_val)])
        
        #예측 정확도
        score = xgb_model.score(X_val, y_val)
        print(f"{name} score:", score)