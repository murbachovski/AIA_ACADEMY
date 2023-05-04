from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. DATA
datasets = fetch_california_housing()
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

x['Population'] = np.log(x['Population'])
# #######################################y log cnage###################
y = np.log1p(y)
#########################

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    train_size=0.8,
    random_state=377
)

# 2. MODEL
model = RandomForestRegressor()

model.fit(x_train, y_train)


#4 .COMPILE
score = model.score(x_test, y_test)
print("SCORE: ", score)

r2_score(np.expm1(y_test), np.expm1(model.predict(y_test)))
print("R2: ", r2_score)

#LOG CHANGE
# 전 SCORE:  0.8166830016159595
# 후 SCORE:  0.817400729762068

# SCORE:  0.8393462472079883