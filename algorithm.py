from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import combinations
from datetime import datetime, timedelta
import pandas as pd

# =====================================
# calculate_date_range(input_date, n)
# =====================================
# This function calculates the date range based on the input date and the specified number of months (n)
# =====================================
# The function assumes that the input_date parameter is a datetime object. 
# If it is a string, you will need to uncomment the line that converts the input date to a datetime object using the datetime.strptime() function.
# -------------------------------------

def calculate_date_range(input_date, n):
    date_format = "%Y-%m-%d"  # Date format
    # input_date = datetime.strptime(date, date_format)  # Convert the input date to a datetime object

    if n == 1:
        # Calculate the range for the previous 30 days
        start_date_30 = (input_date - timedelta(days=30)).strftime(date_format)
        end_date_30 = (input_date - timedelta(days=1)).strftime(date_format)
        range_30 = [start_date_30, end_date_30]
        return range_30

    elif n == 2:
        # Calculate the range from 31 to 60 days ago
        start_date_31_60 = (input_date - timedelta(days=60)).strftime(date_format)
        end_date_31_60 = (input_date - timedelta(days=31)).strftime(date_format)
        range_31_60 = [start_date_31_60, end_date_31_60]
        return range_31_60

    elif n == 3:
        # Calculate the range from 61 to 90 days ago
        start_date_61_90 = (input_date - timedelta(days=90)).strftime(date_format)
        end_date_61_90 = (input_date - timedelta(days=61)).strftime(date_format)
        range_61_90 = [start_date_61_90, end_date_61_90]
        return range_61_90
    
    elif n == 4:
        # Calculate the range from 91 to 120 days ago
        start_date_91_120 = (input_date - timedelta(days=120)).strftime(date_format)
        end_date_91_120 = (input_date - timedelta(days=91)).strftime(date_format)
        range_91_120 = [start_date_91_120, end_date_91_120]
        return range_91_120
    
    elif n == 5:
         # Calculate the range from 121 to 150 days ago
        start_date_121_150 = (input_date - timedelta(days=150)).strftime(date_format)
        end_date_121_150 = (input_date - timedelta(days=121)).strftime(date_format)
        range_121_150 = [start_date_121_150, end_date_121_150]
        return range_121_150

    elif n == 6:
         # Calculate the range from 151 to 180 days ago
        start_date_151_180 = (input_date - timedelta(days=180)).strftime(date_format)
        end_date_151_180 = (input_date - timedelta(days=151)).strftime(date_format)
        range_151_180 = [start_date_151_180, end_date_151_180]
        return range_151_180

    return False


# =========================================
# add_previous_feature(df_list, name_list)
# =========================================
# It adds several previous features to each dataframe based on a specified number of previous months.
# The modified dataframes are saved as CSV files with the corresponding names.
# =========================================
def add_previous_feature(df_list, name_list):
    date_format = "%Y-%m-%d"  # Date format

    for df, name in zip(df_list, name_list):
        if name == '마늘':
            continue

        for idx in df.index:
            df.loc[idx, '일시'] = datetime.strptime(df.loc[idx, '일시'], date_format)

        for n in range(2, 7):
            print(str(n) + "----------------")

            # Initialize new columns
            df['직전 '+str(n)+'달 수출(달러)'] = -1  
            df['직전 '+str(n)+'달 수입(달러)'] = -1
            df['직전 '+str(n)+'달 평균기온(°C)'] = -1  
            df['직전 '+str(n)+'달 평균 상대습도(%)'] = -1
            df['직전 '+str(n)+'달 합계 일사량(MJ/m2)'] = -1  
            df['직전 '+str(n)+'달 평균 풍속(m/s)'] = -1
            df['직전 '+str(n)+'달 최대 풍속(m/s)'] = -1  
            df['직전 '+str(n)+'달 최고기온(°C)'] = -1
            df['직전 '+str(n)+'달 최저기온(°C)'] = -1  
            df['직전 '+str(n)+'달 평균 지면온도(°C)'] = -1
            
            
            # 1) Calculate the total exports (in dollars) for the previous n months.
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for item in in_range_df['수출(달러)'].unique():
                    tmp_df = in_range_df[in_range_df['수출(달러)'] == item] 
                    tmp_list.append(item*(len(tmp_df)/30))

                df.loc[idx, '직전 '+str(n)+'달 수출(달러)'] = sum(tmp_list)

            # 2) Calculate the total imports (in dollars) for the previous n month.     
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for item in in_range_df['수입(달러)'].unique():
                    tmp_df = in_range_df[in_range_df['수입(달러)'] == item] 
                    tmp_list.append(item*(len(tmp_df)/30))

                df.loc[idx, '직전 '+str(n)+'달 수입(달러)'] = sum(tmp_list)

            # 3) Calculate the average temperature for the previous n month      
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for tmp in in_range_df['평균기온(°C)']:
                    tmp_list.append(tmp)

                if tmp_list != []:
                    df.loc[idx, '직전 '+str(n)+'달 평균기온(°C)'] = sum(tmp_list)/len(tmp_list)
                

            # 4) Calculate the average relative humidity for the previous n months. 
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for tmp in in_range_df['평균 상대습도(%)']:
                    tmp_list.append(tmp)

                if tmp_list != []:
                    df.loc[idx, '직전 '+str(n)+'달 평균 상대습도(%)'] = sum(tmp_list)/len(tmp_list)

            # 5) Calculate the total solar radiation for the previous n months.        
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for tmp in in_range_df['합계 일사량(MJ/m2)']:
                    tmp_list.append(tmp)

                df.loc[idx, '직전 '+str(n)+'달 합계 일사량(MJ/m2)'] = sum(tmp_list)

            # 6) Calculate the average wind speed for the previous n months.      
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for tmp in in_range_df['평균 풍속(m/s)']:
                    tmp_list.append(tmp)

                if tmp_list != []:
                    df.loc[idx, '직전 '+str(n)+'달 평균 풍속(m/s)'] = sum(tmp_list)/len(tmp_list)
            
            # 7) Calculate the maximum wind speed for the previous n months.     
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                best = 0
                for tmp in in_range_df['최대 풍속(m/s)']:
                    if tmp > best:
                        best = tmp

                df.loc[idx, '직전 '+str(n)+'달 최대 풍속(m/s)'] = best

            # 8) Calculate the highest temperature for the previous n months.
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                best = 0
                for tmp in in_range_df['최고기온(°C)']:
                    if tmp > best:
                        best = tmp

                df.loc[idx, '직전 '+str(n)+'달 최고기온(°C)'] = best

            # 9) Calculate the lowest temperature for the previous n months.
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                best = 0
                for tmp in in_range_df['최저기온(°C)']:
                    if tmp < best:
                        best = tmp

                df.loc[idx, '직전 '+str(n)+'달 최저기온(°C)'] = best

            # 10) Calculate the average ground temperature for the previous n months.
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue
                
                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for tmp in in_range_df['평균 지면온도(°C)']:
                    tmp_list.append(tmp)

                if tmp_list != []:
                    df.loc[idx, '직전 '+str(n)+'달 평균 지면온도(°C)'] = sum(tmp_list)/len(tmp_list)

        df.to_csv("add_previous_feature/"+name+"_df.csv")
        
# ===========================================================================
# find_best_feature_combination(df_list, item_list, feature_list, train_size)
# ===========================================================================
# It performs feature selection by trying different combinations of features 
# and evaluating their performance using linear regression.
# ===========================================================================
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

        print("best error: ", min(error))
        print("best combination: ", comb[error.index(min(error))])
        
        best_combination.append(comb[error.index(min(error))])

    return best_combination

# ========================================
# multipleRegression(df, item, train_size)
# ========================================
# The function fits a linear regression model, calculates the mean squared error (MSE) 
# for both the training and test sets, and prints the results. 
# It also returns the rounded coefficients of the regression equation and 
# the trained linear regression model.
# ========================================
def multipleRegression(df, item, train_size):

    x = df.drop('인플레이션 반영가', axis=1)
    y = df['인플레이션 반영가']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=(1-train_size), shuffle=True, random_state=1)

    # print("x: ", x)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    y_train_predict = mlr.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_predict)

    y_test_predict = mlr.predict(x_test)
    test_mse = mean_squared_error(y_test, y_test_predict)

    print("Train MSE: ", train_mse)
    print("Test MSE: ", test_mse)

    coefficients = mlr.coef_
    coefficients_rounded = [round(coef, 2) for coef in coefficients]
    print(coefficients_rounded)
    print("회귀식의 계수: ", coefficients_rounded)

    # Plot the scatter plot of predicted vs. actual values with regression line
    plt.scatter(y_test, y_test_predict, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Regression Line')
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("MULTIPLE LINEAR REGRESSION - " + item)
    plt.legend()
    plt.show()

    return coefficients_rounded, mlr

# ===========================================================
# linearRegression(df, target_df, item, train_size, col_name)
# ===========================================================
# The function fits a linear regression model, calculates the mean squared error (MSE) 
# between the predicted and actual values, and prints the error.
# ========================================
def linearRegression(df, target_df, item, train_size, col_name):

    x = df.to_frame()
    y = target_df

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=(1-train_size), shuffle=True, random_state=1)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    print("error: ", mse)

    # Plot the scatter plot of predicted vs. actual values with regression line
    plt.scatter(y_test, y_predict, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Regression Line')
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("MULTIPLE LINEAR REGRESSION - " + item + " feature - " + col_name) 
    plt.legend()
    plt.show()

# visualize regression line
def plot_regression_line(x, y, y_predict):

    # Plot the scatter plot of predicted vs. actual values
    plt.scatter(x, y, alpha=0.4, label='Actual')
    plt.scatter(x, y_predict, alpha=0.4, label='Predicted')
    
def visualizeDistribution(df_list, feature_name):
    for df in df_list:
        plt.plot(df[feature_name])
        plt.show()

