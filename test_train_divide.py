import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def divideData(data):
    X = data.drop("인플레이션 반영가", axis=1)
    Y = data["인플레이션 반영가"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=777)
    return x_train, x_test, y_train, y_test