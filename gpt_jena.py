import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('./_data/kaggle_jena/jena_climate_2009_2016.csv')
df = df[['Date Time', 'T (degC)']]  # Select the date time and temperature columns
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')  # Convert to datetime format
df = df.set_index('Date Time')

# Split the dataset into train and test sets
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Normalize the dataset
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# Define the number of time steps to use in each training sequence
look_back = 60

# Create sequences of input data and corresponding output values
def create_sequences(data):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[(i+look_back), 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(test_scaled)

# Reshape the input data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Make predictions using the model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Invert the scaling of predictions and actual values
train_pred = scaler.inverse_transform(train_pred)
y_train = scaler.inverse_transform([y_train])
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE score for train and test sets
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_pred[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_pred[:,0]))
print("Train RMSE: {:.2f}".format(train_rmse))
print("Test RMSE: {:.2f}".format(test_rmse))
