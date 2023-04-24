import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



#1. DATE
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv =  pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

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
    
# 8
# BEFORE_PCA
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.2895771892656308

# 7
# AFTER_PCA
# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.2938420048759155