import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#연도 수입(달러), 평균기온, 최저기온, 최고기온
def feature_pepper(data):
    data.drop(["일시", "품목", "최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)","수출(kg)","수출(달러)","수입(kg)", "월", "일"], axis=1, inplace=True)

#연도 수출(kg), 최저기온, 평균 상대습도
def feature_radish(data):
    data.drop(["일시", "품목", "평균기온(°C)", "최고기온(°C)", "최소 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","평균 지면온도(°C)","수출(달러)","수입(kg)","수입(달러)", "월", "일"], axis=1, inplace=True)

#연도 평균 지면온도, 평균기온, 최저기온, 최고기온
def feature_cabbage(data):
    data.drop(["일시", "품목", "최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","수출(kg)","수출(달러)","수입(kg)","수입(달러)", "월", "일"], axis=1, inplace=True)

#연도, 수입(달러), 수출(kg)
def feature_garlic(data):
    data.drop(["평균기온(°C)","최저기온(°C)","최고기온(°C)", "일시", "품목", "최소 상대습도(%)","평균 상대습도(%)","최대 풍속(m/s)",
               "평균 풍속(m/s)","합계 일사량(MJ/m2)","합계 일조시간(hr)","수출(달러)","수입(kg)","평균 지면온도(°C)", "월", "일"], axis=1, inplace=True)