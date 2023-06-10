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
    target = '인플레이션 반영가'

    for i, (df, target_df) in enumerate(zip(df_list, target_list)):
        print(df.columns)
        scaler = StandardScaler()
        # Y = scaler.fit_transform(df.iloc[:, 1:])

        x = df.drop(target, axis=1)
        y = df[target]
        x_scaled = scaler.fit_transform(x.iloc[:, 1:])

        # Perform PCA
        pca = PCA(n_components=6)  # Specify the desired number of components
        transformed_data = pca.fit_transform(x_scaled)

        # Create a new dataframe with the transformed data
        transformed_df = pd.DataFrame(transformed_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
        
        transformed_df['가격'] = y.reset_index(drop=True)
        
        # Update the dataframe in the df_list
        df_list[i] = transformed_df

    return df_list

def bagging_multiple_regression(df_list, name_list):
    # Preprocess the data
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    # Apply PCA
    df_list = apply_PCA_values(df_list, target_list)
    
    # Separate target and PCA components
    for i in range(len(df_list)):
        target_list[i] = df_list[i]['가격']
        df_list[i] = df_list[i].drop('가격', axis=1)
    
    for df, target, name in zip(df_list, target_list, name_list):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

        # Create the base classifier
        base_classifier = LinearRegression()

        # Create the Bagging regressor
        bagging_regression = BaggingRegressor(base_classifier, n_estimators=10, random_state=42)

        # Train the Bagging regressor
        bagging_regression.fit(X_train, y_train)

        # Predict with the Bagging regressor
        y_pred = bagging_regression.predict(X_test)

        # Evaluate accuracy (R2 score)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} multiple r2 score:", r2)

def bagging_polynomial_regression(df_list, name_list, degree):
    x_poly_list = []
    
    # Preprocess the data
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    
    for i in range(len(df_list)):
        x_poly_list.append(poly_features.fit_transform(df_list[i]))
    
    for x_poly, target, name in zip(x_poly_list, target_list, name_list):

        # Create the base classifier
        base_classifier = LinearRegression()

        # Create the Bagging regressor
        bagging_regression = BaggingRegressor(base_classifier, n_estimators=10, random_state=42)

        # Train the Bagging regressor
        bagging_regression.fit(x_poly, target)

        # Predict with the Bagging regressor
        y_pred = bagging_regression.predict(x_poly).reshape(-1, 1)

        # Evaluate accuracy (R2 score)
        r2 = r2_score(target, y_pred)
        print(f"{name} polynomial r2 score:", r2)

def XGB_multiple_regression(df_list, name_list):
    
    # Preprocess the data
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    
    for df, target, name in zip(df_list, target_list, name_list):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
        
        # Create XGBoost object
        xgb = XGBRegressor(n_estimators=100, max_depth=3)
        
        # Create parameter grid
        xgb_param_grid = {
            'n_estimators' : [100, 200, 300],
            'learning_rate' : [0.01, 0.1, 0.15, 0.2],
            'max_depth' : [3, 4, 5]
        }
        
        evals = [(X_test, y_test)]
        
        # Create the model
        xgb_grid = GridSearchCV(xgb, param_grid=xgb_param_grid, scoring='r2', n_jobs=1, verbose=1)
        
        # Train the model
        xgb_grid.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=evals, verbose=1)
        
        # Save and load the model
        joblib.dump(xgb_grid, 'xgboost/multiple_r.pkl')
        loaded_model = joblib.load('xgboost/multiple_r.pkl')
        
        print('Best mean accuracy: {:.4f}'.format(loaded_model.best_score_))
        print('Best params: ', loaded_model.best_params_)

def XGB_polynomial_regression(df_list, name_list, degree):
    x_poly_list = []
    
    # Preprocess the data
    df_list, target_list = drop_unusable_feature(df_list, name_list)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    
    for i in range(len(df_list)):
        x_poly_list.append(poly_features.fit_transform(df_list[i]))
    
    for x_poly, target, name in zip(x_poly_list, target_list, name_list):
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x_poly, target, test_size=0.2, random_state=42)
        
        # Split the data into test and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=12)
        
        # Create the model
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
        xgb_model = xgb_model.fit(X_train, y_train, early_stopping_rounds=100,
                              eval_metric='logloss', eval_set=[(X_val, y_val)])
        
        # Evaluate accuracy (R2 score)
        score = xgb_model.score(X_val, y_val)
        print(f"{name} score:", score)
