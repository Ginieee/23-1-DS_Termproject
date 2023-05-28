import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setting_exploration(data_list):
    for data in data_list:
        data.drop("Unnamed: 0", axis=1, inplace=True)

def data_exploration(data_list, data_name_list):
    for data, name in zip(data_list, data_name_list):
        print(f"================================ {name} ================================")
        # Print dataset statistical data
        print(f"----------------------- statistical data({name}) -----------------------")
        print(data.describe())
        print()

        # Print Feature names
        print(f"--------------------- feature names({name}) --------------------")
        print(data.columns.values)
        print()

        # Print data types
        print(f"------------------- feature data types({name}) -------------------")
        print(data.dtypes)
        print()

        #Print number of null value
        print(f"------------------- number of null value({name}) -------------------")
        print(data.isna().sum())
        print()