import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def standardScale(df_list):
    scaler = StandardScaler()
    for df in df_list:
        df_normarlize = scaler.fit_transform(df)
        df[:] = df_normarlize