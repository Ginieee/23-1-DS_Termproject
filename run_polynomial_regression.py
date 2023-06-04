from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# polynomial regression
def polynomial_regression(df, target, degree):
    # Prepare the input features (x) and target variable (y)
    tmp_df = df.copy()
    tmp_df = tmp_df.drop(target, axis=1)
    x = tmp_df
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


def run_polynomial_regression(df_list, target, degree):
    for df in df_list:
        polynomial_regression(df, target, degree)