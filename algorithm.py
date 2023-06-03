from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import combinations
from datetime import datetime, timedelta
import pandas as pd


def calculate_date_range(input_date, n):
    date_format = "%Y-%m-%d"  # 입력된 날짜 형식
    # input_date = datetime.strptime(date, date_format)  # 입력된 날짜를 datetime 객체로 변환

    if n == 1:
        # 직전 30일의 범위 계산
        start_date_30 = (input_date - timedelta(days=30)).strftime(date_format)
        end_date_30 = (input_date - timedelta(days=1)).strftime(date_format)
        range_30 = [start_date_30, end_date_30]
        return range_30

    elif n == 2:
        # 직전 31일부터 60일의 범위 계산
        start_date_31_60 = (input_date - timedelta(days=60)).strftime(date_format)
        end_date_31_60 = (input_date - timedelta(days=31)).strftime(date_format)
        range_31_60 = [start_date_31_60, end_date_31_60]
        return range_31_60

    elif n == 3:
        # 직전 61일부터 90일의 범위 계산
        start_date_61_90 = (input_date - timedelta(days=90)).strftime(date_format)
        end_date_61_90 = (input_date - timedelta(days=61)).strftime(date_format)
        range_61_90 = [start_date_61_90, end_date_61_90]
        return range_61_90
    
    elif n == 4:
        # 직전 91일부터 120일의 범위 계산
        start_date_91_120 = (input_date - timedelta(days=120)).strftime(date_format)
        end_date_91_120 = (input_date - timedelta(days=91)).strftime(date_format)
        range_91_120 = [start_date_91_120, end_date_91_120]
        return range_91_120
    
    elif n == 5:
         # 직전 121일부터 150일의 범위 계산
        start_date_121_150 = (input_date - timedelta(days=150)).strftime(date_format)
        end_date_121_150 = (input_date - timedelta(days=121)).strftime(date_format)
        range_121_150 = [start_date_121_150, end_date_121_150]
        return range_121_150

    elif n == 6:
         # 직전 151일부터 180일의 범위 계산
        start_date_151_180 = (input_date - timedelta(days=180)).strftime(date_format)
        end_date_151_180 = (input_date - timedelta(days=151)).strftime(date_format)
        range_151_180 = [start_date_151_180, end_date_151_180]
        return range_151_180

    return False

  
def add_previous_feature(df_list, name_list):
    date_format = "%Y-%m-%d"  # 날짜 형식

    for df, name in zip(df_list, name_list):

        for idx in df.index:
                df.loc[idx, '일시'] = datetime.strptime(df.loc[idx, '일시'], date_format)

        for n in range(1, 7):
            print(str(n) + "----------------")

            # 새로운 열 초기화
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
            
            # 1) 직전 n달 총 수출(달러) 계산
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

            # 2) 직전 1달 총 수입(달러) 계산        
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

            # 3) 직전 1달 평균 기온 계산        
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
                

            # 4) 직전 1달 평균 상대습도 계산        
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

            # 5) 직전 1달 합계 일사량 계산        
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

            # 6) 직전 1달 평균 풍속 계산        
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
            
            # 7) 직전 1달 최대 풍속 계산        
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

            # 8) 직전 1달 최고 기온 계산
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

            # 9) 직전 1달 최저 기온 계산
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

            # 10) 직전 1달 평균 지면 온도 계산
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
                # mlr.append(mlr)

        print("best error: ", min(error))
        print("best combination: ", comb[error.index(min(error))])
        
        best_combination.append(comb[error.index(min(error))])

    return best_combination

                    # plt.scatter(y_test, y_predict, alpha=0.4)
                    # plt.xlabel("Actual value")
                    # plt.ylabel("Predicted value")
                    # plt.title("MULTIPLE LINEAR REGRESSION - " + item + " - Features: " + ', '.join(combination))
                    # plt.show(


def run_multipleRegression(df_list, item_list, combination_list, train_size):
    for df, item, feature in zip(df_list, item_list, combination_list):
        x = df[feature]
        y = df['인플레이션 반영가']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=(1-train_size), shuffle=True, random_state=1)

        mlr = LinearRegression()
        mlr.fit(x_train, y_train) 

        y_predict = mlr.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        print("error: ", mse)

        plt.scatter(y_test, y_predict, alpha=0.4)
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.title("MULTIPLE LINEAR REGRESSION - " + item + " - Features: " + ', '.join(feature))
        plt.axis('equal')
        plt.show()
        
    
def visualizeDistribution(df_list, feature_name):
    for df in df_list:
        plt.plot(df[feature_name])
        plt.show()

