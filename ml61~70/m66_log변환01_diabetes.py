from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. DATA
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
# print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

# df['Population'].plot.box()
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

# ##################################x_Populataion log change####################
# df['Population']
y = df['target']
x = df.drop(['target'], axis=1)

x = np.log1p(x)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    train_size=0.8,
    random_state=377
)
# # #######################################y log cnage###################
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
# # #######################################y log cnage###################

# 2. MODEL
model = RandomForestRegressor()

model.fit(x_train, y_train_log)


#4 .COMPILE
score = model.score(x_test, y_test_log)
print("SCORE: ", score)

r2_score(y_test, np.expm1(model.predict(x_test)))
print("R2: ", r2_score)

#LOG CHANGE
# SCORE:  0.3616742143279448
# R2:  <function r2_score at 0x000002673EC8E4C0>