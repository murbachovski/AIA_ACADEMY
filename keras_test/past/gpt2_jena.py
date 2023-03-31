import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# load the data
df = pd.read_csv('./_data/kaggle_jena/jena_climate_2009_2016.csv')

# extract the temperature data
temp = df['T (degC)'].values.reshape(-1, 1)

# normalize the temperature data
scaler = StandardScaler()
temp = scaler.fit_transform(temp)

# split the data into training and testing sets
train_size = int(len(temp) * 0.8)
test_size = len(temp) - train_size
train, test = temp[0:train_size,:], temp[train_size:len(temp),:]

# function to create a dataset with look back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# create the dataset with look back
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape the data for LSTM input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(trainX, trainY, epochs=10, batch_size=64, verbose=2)

# make predictions on test data
testPredict = model.predict(testX)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(testY, testPredict))
print('Test RMSE: %.3f' % rmse)
