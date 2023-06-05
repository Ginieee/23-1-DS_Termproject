from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import combinations
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score
        
# feature리스트에서 error가 최소가 되는 최적의 조합을 찾는 함수
def find_best_feature_combination(df_list, item_list, feature_list, train_size):
    plt.clf()
    garlic_error_list = []
    cabbage_error_list = []
    radish_error_list = []
    pepper_error_list = []
    
    error_list = [garlic_error_list, cabbage_error_list, radish_error_list, pepper_error_list]

    garlic_comb_list = []
    cabbage_comb_list = []
    radish_comb_list = []
    pepper_comb_list = []

    comb_list = [garlic_comb_list, cabbage_comb_list, radish_comb_list, pepper_comb_list]

    garlic_mlr_list = []
    cabbage_mlr_list = []
    radish_mlr_list = []
    pepper_mlr_list = []

    mlr_list = [garlic_mlr_list, cabbage_mlr_list, radish_mlr_list, pepper_mlr_list]

    best_combination = []

    for df, item, error, comb, mlr in zip(df_list, item_list, error_list, comb_list, mlr_list):
        df = df[df['연도'] != 1999]
        print("[" + item + "]")
        
        for r in range(1, len(feature_list) + 1):
            feature_combinations = combinations(feature_list, r)

            for combination in feature_combinations:
                x = df[list(combination)]
                y = df['인플레이션 반영가']
                
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=(1-train_size), shuffle=True, random_state=1)
        
                mlr = LinearRegression()
                mlr.fit(x_train, y_train) 

                y_predict = mlr.predict(x_test)
                mse = mean_squared_error(y_test, y_predict)

                error.append(mse)
                comb.append(list(combination))
                # mlr.append(mlr)

        print("best error: ", min(error))
        print("best combination: ", comb[error.index(min(error))])
        
        best_combination.append(comb[error.index(min(error))])

    return best_combination

# multiple linear regression을 돌리는 함수
def multipleRegression(df, item, train_size):

    x = df.drop('가격', axis=1)
    y = df['가격']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=(1-train_size), shuffle=True, random_state=1)

    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    y_predict = mlr.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print("error: ", mse)
    print(f"{item} r2 : ", r2)

    # Plot the scatter plot of predicted vs. actual values with regression line
    plt.scatter(y_test, y_predict, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Regression Line')
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("MULTIPLE LINEAR REGRESSION - " + item)
    plt.legend()
    plt.show()

# regression line을 그려주는 함수
def plot_regression_line(x, y, y_predict):
    # Plot the scatter plot of predicted vs. actual values
    plt.scatter(x, y, alpha=0.4, label='Actual')
    plt.scatter(x, y_predict, alpha=0.4, label='Predicted')
    
    # Sort the values in ascending order for a smooth line plot
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_y_predict = y_predict[sorted_indices]
    
    # Plot the regression line
    plt.plot(sorted_x, sorted_y_predict, color='red', label='Regression Line')
    
    plt.xlabel("Input features")
    plt.ylabel("Target variable")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.show()

# polynomial regression
def polynomial_regression(df, column_list, target, degree):
    # Prepare the input features (x) and target variable (y)
    x = df[column_list]
    y = df[target]
    
    # Apply polynomial transformation to the input features
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

    print('x_poly: ', x_poly)
    
    # Train a linear regression model with the polynomial features
    model = LinearRegression()
    model.fit(x_poly, y)
    
    # Predict the target variable using the trained model
    y_predict = model.predict(x_poly)
    
    # Compute and print the score (R-squared) of the model
    print('score: ', model.score(x_poly, y))    
    
    # Plot the scatter plot of predicted vs. actual values
    plt.scatter(y, y_predict, alpha=0.4)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Scatter Plot - Actual vs. Predicted")
    plt.show()

# 분포를 표시하는 함수
def visualizeDistribution(df_list, feature_name):
    for df in df_list:
        plt.plot(df[feature_name])
        plt.show()

# polynomial regression을 테스트하는 함수
def test_polynomial_regression(poly_features, model, df, column_list):
    # Prepare the input features (x)
    x = df[column_list]
    
    # Apply polynomial transformation to the input features
    x_poly = poly_features.transform(x)

    # Predict the target variable using the trained model
    y_predict = model.predict(x_poly)
    
    # Compare y_predict and y_actual
    y_actual = df['인플레이션 반영가']
    comparison = pd.DataFrame({'y_predict': y_predict, 'y_actual': y_actual})
    print(comparison)

    # Print y_predict and y_actual separately
    print('y_predict:', y_predict)
    print('y_actual:', y_actual)



