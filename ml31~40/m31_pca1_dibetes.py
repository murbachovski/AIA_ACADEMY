import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#1. DATA
dataset = load_diabetes()

x = dataset['data']
y = dataset.target
# print(x.shape, y.shape) # (442, 10) (442,)

# MODEL
pca = PCA(n_components=9)
x = pca.fit_transform(x)
# print(x.shape) # (442, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=123
)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123, n_jobs=-1)

# COMPILE
model.fit(x_train, y_train)

# PREDICT
results = model.score(x_test, y_test)
print("RESULTS :", results)
# BEFORE PCA 
# RESULTS : 0.47283290964179814

# AFTER PCA
# RESULTS : 0.4598408858866415