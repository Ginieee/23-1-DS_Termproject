import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def kMeans(df_list, name_list):
    fig, ax = plt.subplots(1, 4)
    for df, name, index in zip(df_list, name_list, range(4)):
        
        
        #두개의 PCA로 feature 묶음
        pca = PCA(n_components=2)
        pca.fit(df)
        df_pca = pca.transform(df)
        df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
        
        #KMeans
        kmeans = KMeans(n_clusters=3)
        #centroids = kmeans.cluster_centers_
        kmeans.fit(df_pca)
        ax[index].scatter(df_pca['PCA1'], df_pca['PCA2'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        #ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
        ax[index].set_xlabel('PCA1')
        ax[index].set_ylabel('PCA2')
        ax[index].set_title(f'Cluser of {name}')