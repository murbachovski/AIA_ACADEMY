import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. DATA
datasets = [
    load_breast_cancer()
]
after = False
for i, v in enumerate(datasets):
    x, y = v.data, v.target
    print(x.shape, y.shape) # (569, 30) (569,)
    if after == True:
        n_pca = x.shape[1] - 1
    else:
        n_pca = x.shape[1]
        # MODEL
        pca = PCA(n_components=n_pca)
        x = pca.fit_transform(x)
        print(n_pca)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=123
        )
        model = RandomForestRegressor(random_state=123, n_jobs=-1)

        # COMPILE
        model.fit(x_train, y_train)

        # PREDICT
        results = model.score(x_test, y_test)
        print('model_name: ', model)
        print("RESULTS :", results)
    
# BEFORE PCA 
# 30
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.8699226862679585

# AFTER PCA -1
#   29
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.8695684597393919