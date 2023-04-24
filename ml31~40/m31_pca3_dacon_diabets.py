import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. DATA
datasets = [
    load_diabetes(),
    load_breast_cancer()
]

for i, v in enumerate(datasets):
    x, y = v.data, v.target
    print(x.shape, y.shape) # (442, 10) (442,)
    if x.shape[1] == 10:
        n_pca = 9
        if x.shape[1] == 9:
            n_pca = 8

        # MODEL
        pca = PCA(n_components=n_pca)
        x = pca.fit_transform(x)
        
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
    # RESULTS : 0.47283290964179814

    # AFTER PCA
    # RESULTS : 0.4598408858866415