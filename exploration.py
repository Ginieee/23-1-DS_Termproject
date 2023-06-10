import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setting_exploration(data_list):
    for data in data_list:
        try:
            data.drop("Unnamed: 0", axis=1, inplace=True)
            data.drop("Unnamed: 0.1", axis=1, inplace=True)
        except KeyError:
            pass
        pd.set_option('display.expand_frame_repr', False)

def data_exploration(data_list, data_name_list):
    for data, name in zip(data_list, data_name_list):
        # print(f"================================ {name} ================================")
        # #Print head of data
        # print(f"----------------------- data head({name}) -----------------------")
        # print()
        # print(data.head())
        # print()
        
        # # Print dataset statistical data
        # print(f"----------------------- statistical data({name}) -----------------------")
        # print()
        # print(data.describe())
        # print()

        # Print Feature names
        print(f"--------------------- feature names({name}) --------------------")
        print()
        print(data.columns.values)
        print()

        # # Print data types
        # print(f"------------------- feature data types({name}) -------------------")
        # print()
        # print(data.dtypes)
        # print()

        # #Print number of null value
        # print(f"------------------- number of null value({name}) -------------------")
        # print()
        # print(data.isna().sum())
        # print()
        
        # #Print number of row, column
        # print(f"------------------- number of row, column({name}) -------------------")
        # print()
        # print("Row count:", data.shape[0])
        # print("Column count:", data.shape[1])
        # print()
        
        #print dirty value
        # print(f"------------------- number of dirty data({name}) -------------------")
        # print()
        # count=0
        # for i, row in data.iterrows():
        #     # 1: humidty over 100% : count++
        #     if row['평균 상대습도(%)'] > 100:
        #         count += 1
        #         continue

        #     # 2: reverse temperature : count++
        #     if row['최저기온(°C)'] > row['최고기온(°C)']:
        #         count += 1
        #         continue

        #     # 3: not in item : count++
        #     if row['품목'] not in item:
        #         count += 1
        #         continue

        #     # 4: year >= 2024 : count++
        #     if row['연도'] >= 2024:
        #         count += 1
        #         continue
            
        #     # 5: import == 0  & export == 0 : count++
        #     if row['수출(kg)']==0 and row['수입(kg)']==0:
        #         count += 1
        #         continue
            
        # print('count: ', count)