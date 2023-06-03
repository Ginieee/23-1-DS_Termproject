import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def dataDropna(data):
    data = data.dropna(axis=0, inplace=True)

def multiple_regression(x_train_list, x_test_list, y_train_list, y_test_list, name_list):
    fig, ax = plt.subplots(1, 4)
    for x_train, x_test, y_train, y_test, name, index in zip(x_train_list, x_test_list, y_train_list, y_test_list, name_list, range(4)):
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        model_predict = model.predict(x_test)
        
        ax[index].scatter(y_test, model_predict, alpha=0.4)
        ax[index].set_xlabel("Actual Price")
        ax[index].set_ylabel("Predict Price")
        ax[index].set_title(f"Multiple Regression : {name}")
        
        #print model score
        print(f"{name} 정확도: {model.score(x_train, y_train)}")