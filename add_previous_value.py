from datetime import datetime, timedelta
import numpy as np

# 입력된 날짜로부터 n달의 범위를 리턴해주는 함수
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

# 직전 feature 추가하는 함수
def add_previous_feature(df_list, name_list):
    date_format = "%Y-%m-%d"  # 날짜 형식

    result_df = []

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
        
        result_df.append(df)

        # df.to_csv("add_previous_feature/"+name+"_df.csv")
    return result_df
                
# 직전 가격 feature를 추가하는 함수
def add_previous_price_feature(df_list):
    date_format = "%Y-%m-%d"  # 날짜 형식

    result_df_list = []

    for df in df_list:

        for idx in df.index:
                words = df.loc[idx, '일시'].split()
                df.loc[idx, '일시'] = words[0]
                df.loc[idx, '일시'] = datetime.strptime(df.loc[idx, '일시'], date_format)

        for n in range(2, 7):
            print(str(n) + "----------------")

            # 새로운 열 초기화
            df['직전 '+str(n)+'달 인플레이션 반영가'] = np.nan
            
            for idx in df.index:
                if df.loc[idx, '연도'] == 1999:
                    continue

                range_date = calculate_date_range(df.loc[idx, '일시'], n)

                start_date, end_date = datetime.strptime(range_date[0], date_format), datetime.strptime(range_date[1], date_format)
                
                in_range_df = df[(df['일시'] >= start_date) & (df['일시'] <= end_date)]
                
                tmp_list = []
                
                for item in in_range_df['인플레이션 반영가']:
                    tmp_list.append(item)
                
                if len(tmp_list) == 0:
                    continue

                df.loc[idx, '직전 '+str(n)+'달 인플레이션 반영가'] = sum(tmp_list)/len(tmp_list)
        
        result_df_list.append(df)
    return result_df_list
