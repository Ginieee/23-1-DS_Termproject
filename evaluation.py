from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from run_linear_regression import drop_unusable_feature
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from xgboost import XGBRegressor
from xgboost import plot_importance
import joblib

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

#https://lsjsj92.tistory.com/547 - 로직 설명
#https://stackoverflow.com/questions/52615401/xgboost-core-xgboosterror-need-to-call-fit-beforehand-when-trying-to-predict - 저장하고 불러오는 이유
#사이킷런 래퍼 XGBoost 모듈
def XGB_multiple_regression(df_list, name_list):
    
    #데이터 전처리
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    # #PCA 적용
    # df_list = apply_PCA_values(df_list, target_list)
    
    # #target, pca로 나눔
    # for i in range(len(df_list)):
    #     target_list[i] = df_list[i]['가격']
    #     df_list[i] = df_list[i].drop('가격', axis=1)
    
    for df, target, name in zip(df_list, target_list, name_list):
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
        #학습과 검증 데이터로 나눔
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=12)
        
        #XGBoost 객체 생성
        xgb = XGBRegressor(n_estimators=100, max_depth=3)
        #xgb = xgb.fit(X_train, y_train, early_stopping_rounds=100,
        #                      eval_metric='logloss', eval_set=[(X_val, y_val)])
        
        #파라미터 그리드 생성
        xgb_param_grid = {
            'n_estimators' : [100, 200, 300],
            'learning_rate' : [0.01, 0.1, 0.15, 0.2],
            'max_depth' : [3, 4, 5]
        }
        
        evals = [(X_test, y_test)]
        
        #모델 생성
        xgb_grid = GridSearchCV(xgb, param_grid = xgb_param_grid, scoring='r2', n_jobs=1, verbose=1)
        
        #학습
        xgb_grid.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=evals, verbose=1)
        
        #학습결과를 저장하고 불러옴
        joblib.dump(xgb_grid, 'xgboost/multiple_r.pkl')
        loaded_model = joblib.load('xgboost/multiple_r.pkl')
        
        print('최고 평균 정확도 : {:.4f}'.format(loaded_model.best_score_))
        print('Best param : ', loaded_model.best_params_)

        #dataframe으로 랭킹순보기
        # result_df = pd.DataFrame(xgb_grid.cv_results_)
        # result_df.sort_values(by=['rank_test_score'],inplace=True)

        # #plot
        # result_df[['params','mean_test_score','rank_test_score']].head(10)
        
        #예측 정확도
        # score = xgb.score(X_val, y_val)
        # print(f"{name} score:", score)

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