import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/dacon_diabetes/')
path_save = ('./_save/dacon_diabetes/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(train_csv.shape, test_csv.shape)    # (652, 9) (116, 8)

#1-2 x, y SPLIT
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']   

after = True

print(x.shape, y.shape) # (652, 8) (652,)

if after == True:
    n_pca = x.shape[1] - 1
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
    if after == True:
        print('AFTER')
        print('model_name: ', model)
        print("RESULTS :", results)
elif after == False:
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
    if after == True:
        print('AFTER PCA')
        print('model_name: ', model)
        print("RESULTS :", results)
    
# BEFORE PCA 
# 8
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.2105871552975327

# 7
# AFTER
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.21023224479922598