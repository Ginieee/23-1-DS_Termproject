from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'

# polynomial regression
def polynomial_regression(df, target, degree):
    x = df
    y = target
    
    # Apply polynomial transformation to the input features
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

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
    plt.title("Train Scatter Plot - Actual vs. Predicted")
    plt.axis('equal')
    plt.show()

    return poly_features, model

# polynomial regression을 테스트하는 함수
def test_polynomial_regression(poly_features, model, df, target_df):
    # Prepare the input features (x)
    x = df
    
    # Apply polynomial transformation to the input features
    x_poly = poly_features.transform(x)

    # Predict the target variable using the trained model
    y_predict = model.predict(x_poly)
    
    # Compare y_predict and y_actual
    # y_actual = df['인플레이션 반영가']
    y_actual = target_df
    comparison = pd.DataFrame({'y_predict': y_predict, 'y_actual': y_actual})
    print(comparison)

    # Print y_predict and y_actual separately
    # print('y_predict:', y_predict)
    # print('y_actual:', y_actual)

    # Plot the scatter plot of predicted vs. actual values
    plt.scatter(y_actual, y_predict, alpha=0.4)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Test Scatter Plot - Actual vs. Predicted")
    plt.axis('equal')
    plt.show()


def drop_unusable_feature(df_list, item_list):
    
    garlic_target_df = pd.DataFrame([])
    napa_cabbage_target_df = pd.DataFrame([])
    radish_target_df= pd.DataFrame([])
    pepper_target_df = pd.DataFrame([])

    target_df_list = [garlic_target_df, napa_cabbage_target_df, radish_target_df, pepper_target_df]

    for i, (df, target_df, item) in enumerate(zip(df_list, target_df_list, item_list)):
        start_with = "직전 "
        except_list = ['직전 3달 인플레이션 반영가', "직전 4달 인플레이션 반영가", "직전 5달 인플레이션 반영가","직전 6달 인플레이션 반영가"]
        if item == "건고추" or item == "pepper":
            for n in range(5, 7):
                df = drop_feature_start_with(df, start_with + str(n) + "달", except_list)
        elif item == "배추"or item == "napa_cabbage":
            for n in range(4, 7):
                df = drop_feature_start_with(df, start_with + str(n) + "달", except_list)
        elif item == "무"or item == "radish":
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

    return df_list, target_df_list

def drop_feature_start_with(df, start_with, except_list):
    
    for index, value in df.iteritems(): 

        if start_with in index and index not in except_list:
            df.drop(index, axis=1, inplace = True)

    return df

def run_polynomial_regression(df_list, item_list, target, degree):
    df_list, target_df_list = drop_unusable_feature(df_list, item_list)

    for df, target in zip(df_list, target_df_list):
        idx_80_percent = int(len(df) * 0.8)
        poly_features, model = polynomial_regression(df[:idx_80_percent], target[:idx_80_percent], degree)
        test_polynomial_regression(poly_features, model, df[idx_80_percent:], target[idx_80_percent:])
