import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. DATA
path = './_data/ddarung/' # path ./은 현재 위치
path_save = './_save/ddarung/'
# Column = Header


# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 

# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 

train_csv = train_csv.dropna()


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

after = False

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
        print('AFTER_PCA')
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
    if after == False:
        print('BEFORE_PCA')
        print('model_name: ', model)
        print("RESULTS :", results)
    
# 9
# BEFORE_PCA
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.6936276869466653

# 8
# AFTER
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.6784454950446617