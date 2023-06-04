import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA


def do_multiple_kmeans(df_list, name_list):
    pca_df_list = []
    for df in df_list:
        df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace = True)
        df = df[df['연도'] > 1999]
        # select useful features for pca
        df = df[['수출(달러)', '수입(달러)', '직전 2달 평균기온(°C)', '직전 2달 평균 상대습도(%)', '직전 2달 합계 일사량(MJ/m2)', '직전 2달 평균 풍속(m/s)', '직전 2달 최대 풍속(m/s)', '직전 2달 최고기온(°C)', '직전 2달 최저기온(°C)', '직전 2달 평균 지면온도(°C)', '직전 1달 평균기온(°C)', '직전 1달 평균 상대습도(%)', '직전 1달 합계 일사량(MJ/m2)', '직전 1달 평균 풍속(m/s)', '직전 1달 최대 풍속(m/s)', '직전 1달 최고기온(°C)', '직전 1달 최저기온(°C)', '직전 1달 평균 지면온도(°C)', '인플레이션 반영가']]

        # doing pca
        pca_df = perform_pca(df, '인플레이션 반영가')
        pca_df_list.append(pca_df)

    # doing kmeans clustering algorithm with pca dataframe
    for i, (df, name) in enumerate(zip(pca_df_list, name_list)):
        multiple_kmeans_algorithm(df, name, df.columns[:-1], df.columns[-1])


def multiple_kmeans_algorithm(df, title, column_list, target):
    kmeans = KMeans(n_clusters=3).fit(df) # n_clusters=3 since we expected the price low, middle, high
    centroids = kmeans.cluster_centers_
    pred = kmeans.predict(df)

    # draw the result of kmeans
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[column_list[0]], df[column_list[1]], df[target], c=pred)  # 3D 산점도 그리기
    ax.set_xlabel(column_list[0])
    ax.set_ylabel(column_list[1])
    ax.set_zlabel(target)
    ax.set_title("Multiple KMeans : {}".format(title))


def perform_pca(df, target):
    target_column = df[target]
    target_column.reset_index(drop=True, inplace=True)
    features = df.drop(columns=[target])
    features.reset_index(drop=True, inplace=True)

    pca = PCA(n_components=2) # 2 was determined by the result of calculate_cumulative_variance_ration
    pca_result = pca.fit_transform(features)

    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df[target] = target_column

    # The pca model was not returned.
    # Because the kmeans clustering algorithm was not done as we wanted, so the pca model was no longer needed.
    return pca_df


def calculate_cumulative_variance_ratio(df):
    pca = PCA(n_components=df.shape[1])
    pca.fit(df)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    return cumulative_variance_ratio


def plot_cumulative_variance_ratio(cumulative_variance_ratio):
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained Ratio')
    plt.title('Cumulative Variance Explained Ratio vs. Number of Principal Components')
    plt.show()