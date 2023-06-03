import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def dataDropna(data):
    data = data.dropna(axis=0, inplace=True)

def multiple_regression(x_train_list, x_test_list, y_train_list, y_test_list, name_list):
    for x_train, x_test, y_train, y_test, name in zip(x_train_list, x_test_list, y_train_list, y_test_list, name_list):
        fig, ax = plt.subplots(1, 1)
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        model_predict = model.predict(x_test)
        
        ax.scatter(y_test, model_predict, alpha=0.4)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predict Price")
        ax.set_title(f"Multiple Regression : {name}")
        
        #print model score
        print(f"{name} 정확도: {model.score(x_train, y_train)}")