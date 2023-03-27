import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#1. DATA
path = ('./_data/kaggle_jena/')

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets) # [420551 rows x 14 columns]

# print(datasets.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')
# print(datasets.info()) #결측치 확인 가능하다.
# <class 'pandas.core.frame.DataFrame'>
# Index: 420551 entries, 01.01.2009 00:10:00 to 01.01.2017 00:00:00
# Data columns (total 14 columns):
#  #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   p (mbar)         420551 non-null  float64
#  1   T (degC)         420551 non-null  float64
#  2   Tpot (K)         420551 non-null  float64
#  3   Tdew (degC)      420551 non-null  float64
#  4   rh (%)           420551 non-null  float64
#  5   VPmax (mbar)     420551 non-null  float64
#  6   VPact (mbar)     420551 non-null  float64
#  7   VPdef (mbar)     420551 non-null  float64
#  8   sh (g/kg)        420551 non-null  float64
#  9   H2OC (mmol/mol)  420551 non-null  float64
#  10  rho (g/m**3)     420551 non-null  float64
#  11  wv (m/s)         420551 non-null  float64
#  12  max. wv (m/s)    420551 non-null  float64
#  13  wd (deg)         420551 non-null  float64
# dtypes: float64(14)
# memory usage: 48.1+ MB
# None
# print(datasets.describe())
#             p (mbar)       T (degC)       Tpot (K)    Tdew (degC)         rh (%)   VPmax (mbar)   VPact (mbar)   VPdef (mbar)      sh (g/kg)  H2OC (mmol/mol)   rho (g/m**3)       wv (m/s)  max. wv (m/s)       wd (deg)
# count  420551.000000  420551.000000  420551.000000  420551.000000  420551.000000  420551.000000  420551.000000  420551.000000  420551.000000    420551.000000  420551.000000  420551.000000  420551.000000  420551.000000
# mean      989.212776       9.450147     283.492743       4.955854      76.008259      13.576251       9.533756       4.042412       6.022408         9.640223    1216.062748       1.702224       3.056555     174.743738
# std         8.358481       8.423365       8.504471       6.730674      16.476175       7.739020       4.184164       4.896851       2.656139         4.235395      39.975208      65.446714      69.016932      86.681693
# min       913.600000     -23.010000     250.600000     -25.010000      12.950000       0.950000       0.790000       0.000000       0.500000         0.800000    1059.450000   -9999.000000   -9999.000000       0.000000
# 25%       984.200000       3.360000     277.430000       0.240000      65.210000       7.780000       6.210000       0.870000       3.920000         6.290000    1187.490000       0.990000       1.760000     124.900000
# 50%       989.580000       9.420000     283.470000       5.220000      79.300000      11.820000       8.860000       2.190000       5.590000         8.960000    1213.790000       1.760000       2.960000     198.100000
# 75%       994.720000      15.470000     289.530000      10.070000      89.400000      17.600000      12.350000       5.300000       7.800000        12.490000    1242.770000       2.860000       4.740000     234.100000
# max      1015.350000      37.280000     311.340000      23.110000     100.000000      63.770000      28.320000      46.010000      18.130000        28.820000    1393.540000      28.490000      23.500000     360.000000

# print(datasets['T (degC)'])
# Date Time
# 01.01.2009 00:10:00   -8.02
# 01.01.2009 00:20:00   -8.41
# 01.01.2009 00:30:00   -8.51
# 01.01.2009 00:40:00   -8.31
# 01.01.2009 00:50:00   -8.27
#                        ... 
# 31.12.2016 23:20:00   -4.05
# 31.12.2016 23:30:00   -3.35
# 31.12.2016 23:40:00   -3.16
# 31.12.2016 23:50:00   -4.23
# 01.01.2017 00:00:00   -4.82
# Name: T (degC), Length: 420551, dtype: float64

# print(datasets['T (degC)'].values)          # pandas to numpy(판다스를 넘파이로 만드는 방법.values)
# [-8.02 -8.41 -8.51 ... -3.16 -4.23 -4.82]

# print(datasets['T (degC)'].to_numpy())      # pandas to numpy(판다스를 넘파이로 만드는 방법.to_numpy)
# [-8.02 -8.41 -8.51 ... -3.16 -4.23 -4.82]

# import matplotlib.pyplot as plt
# plt.plot(datasets['T (degC)'].values)
# plt.show() #규칙성이 있는 시계열 '~~~~'

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']
# print(x, y)
# print(x.shape, y.shape) # (420551, 13) (420551,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    shuffle=False
)
x_test, x_predict, y_test, y_predict = train_test_split(
    x,
    y,
    train_size = 0.3,
    shuffle=False
)

timesteps = 6 
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)
x = split_x(x, timesteps)

#
scaler = MinMaxScaler() 
scaler.fit(x)
x = scaler.transform(x)


# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (294385, 13) (126166, 13) (294385,) (126166,)

#2. MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(13, 14)))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=128)

#4. PREDICT
results = model.evaluate(x_test, y_test)
print(results)
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):                #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt = route
rmse = RMSE(y_test, y_predict)             #RMSE 함수 사용
print("rmse: ", rmse)








# def split_xy3(x, timesteps, y):
#     x, y = list(), list()
#     for i in range(len(datasets)):
#         x_end_number = i + timesteps
#         y_end_number = x_end_number + y - 1
        
#         if y_end_number > len(datasets):
#             break
#         tmp_x = datasets[i:x_end_number, :-1]
#         tmp_y = datasets[x_end_number-1:y_end_number, -1]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)
# x, y = split_xy3(datasets, 3, 1)
# print(x, '\n', y)
# print(x.shape)
# print(y.shape)

# x_train_timesteps = 7
# x_test_timesteps = 2
# def split_x_train(x, x_train_timesteps):
#     aaa = []
#     for i in range(len(x) - x_train_timesteps + 1):
#         subset = x[i : (i + x_train_timesteps)]
#         aaa.append(subset)
#     return np.array(aaa)
# x_train = split_x_train(x, x_train_timesteps)
# print(x_train)
# print(x_train.shape)                 # (420545, 7, 13)

# def split_x_test(x, x_test_timesteps):
#     aaa = []
#     for i in range(len(x) - x_test_timesteps + 1):
#         subset = x[i : (i + x_test_timesteps)]
#         aaa.append(subset)
#     return np.array(aaa)
# x_test = split_x_test(x, x_test_timesteps)
# print(x_test)
# print(x_test.shape)                 # (420550, 2, 13)
