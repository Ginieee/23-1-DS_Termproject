import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from correlation import draw_corr_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def pol_regression(x_train_list, x_test_list, y_train_list, y_test_list, name_list):
    fig, axs = plt.subplots(1, 4)
    for x_train, x_test, y_train, y_test, name, index in zip(x_train_list, x_test_list, y_train_list, y_test_list, name_list, range(4)):
        poly_features = PolynomialFeatures(degree=2)
        x_poly = poly_features.fit_transform(x_train)
        x_poly_test = poly_features.fit_transform(x_test)

        model = LinearRegression()
        model.fit(x_poly, y_train)

        y_predict = model.predict(x_poly_test)

        print("{} 정확도: {}".format(name, model.score(x_poly_test, y_test)))

        axs[index].scatter(y_test, y_predict, alpha=0.4)
        axs[index].set_xlabel("Actual Target")
        axs[index].set_ylabel("Predicted Target")
        axs[index].set_title(name)
        
        y_test = np.array(y_test).reshape(-1, 1)  # numpy 배열로 변환하여 reshape
        y_predict = np.array(y_predict).reshape(-1, 1)
        
        #axs[index].plot(y_test, y_predict, c='red')

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
        print(f"{name} 정확도: {model.score(x_test, y_test)}")

def divideData(data):
    X = data.drop("인플레이션 반영가", axis=1)
    Y = data["인플레이션 반영가"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=777)
    return x_train, x_test, y_train, y_test

def standardScale(df_list):
    scaler = StandardScaler()
    for df in df_list:
        df_normarlize = scaler.fit_transform(df)
        df[:] = df_normarlize

def set(df_list):
    for df in df_list:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    for df in df_list:
        df['일시'] = pd.to_datetime(df['일시'])
        df = df[df['일시'].dt.year != 1999]


garlic_df = pd.read_csv("add_previous_feature/마늘_df.csv", low_memory=False)
napa_cabbage_df = pd.read_csv("add_previous_feature/배추_df.csv", low_memory=False)
radish_df = pd.read_csv("add_previous_feature/무_df.csv", low_memory=False)
pepper_df = pd.read_csv("add_previous_feature/건고추_df.csv", low_memory=False)

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

plt.rcParams['font.family'] = 'Malgun Gothic'

set(df_list)

radish_df = radish_df[radish_df['일시'].dt.year != 1999]
garlic_df = garlic_df[garlic_df['일시'].dt.year != 1999]
pepper_df = pepper_df[pepper_df['일시'].dt.year != 1999]
napa_cabbage_df = napa_cabbage_df[napa_cabbage_df['일시'].dt.year != 1999]

pepper_df = pepper_df[ ['직전 6달 평균 상대습도(%)', '직전 6달 평균 풍속(m/s)', '직전 5달 평균 풍속(m/s)', '직전 6달 수입(달러)', '직전 5달 최대 풍속(m/s)', '직전 6달 최고기온(°C)', '인플레이션 반영가']]
garlic_df = garlic_df[['연도', '수입(달러)', '수입(kg)','인플레이션 반영가']]
radish_df = radish_df[['연도', '직전 3달 평균기온(°C)','직전 3달 평균 상대습도(%)','직전 3달 평균 풍속(m/s)', '직전 3달 합계 일사량(MJ/m2)','인플레이션 반영가']]
napa_cabbage_df = napa_cabbage_df[ ['직전 1달 평균기온(°C)', '직전 1달 평균 지면온도(°C)', '평균 지면온도(°C)', '평균기온(°C)', '직전 1달 최고기온(°C)', '최저기온(°C)', '최고기온(°C)', '인플레이션 반영가']]

df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]

standardScale(df_list)

radish_x_train, radish_x_test, radish_y_train, radish_y_test = divideData(radish_df)
cabbage_x_train, cabbage_x_test, cabbage_y_train, cabbage_y_test = divideData(napa_cabbage_df)
garlic_x_train, garlic_x_test, garlic_y_train, garlic_y_test = divideData(garlic_df)
pepper_x_train, pepper_x_test, pepper_y_train, pepper_y_test = divideData(pepper_df)

train_x_list=[radish_x_train, cabbage_x_train, garlic_x_train, pepper_x_train]
test_x_list=[radish_x_test, cabbage_x_test, garlic_x_test, pepper_x_test]
train_y_list=[radish_y_train, cabbage_y_train, garlic_y_train, pepper_y_train]
test_y_list=[radish_y_test, cabbage_y_test, garlic_y_test, pepper_y_test]

multiple_regression(train_x_list, test_x_list, train_y_list, test_y_list, name_list)
print("다항 회귀-------------------------------------------------------")
pol_regression(train_x_list, test_x_list, train_y_list, test_y_list, name_list)
plt.show()