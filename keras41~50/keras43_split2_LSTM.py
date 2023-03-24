import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

dataset = np.array(range(1, 101))
timesteps = 4
x_predict = np.array(range(96, 106))
timesteps_pred = 4

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

def split_pred_x(x_predict, timesteps_pred):
    ccc = []
    for i in range(len(x_predict) - timesteps_pred + 1):
        subset_pred = x_predict[i : (i + timesteps_pred)]
        ccc.append(subset_pred)
    return np.array(ccc)


bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) 
ddd = split_pred_x(x_predict, timesteps_pred)
# print(ddd)
# print(ddd.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
x_predict = ddd[:, :-1]
print(x.shape)
print(y.shape)
print(x_predict)
print(x_predict.shape)

#2. MODEL
model = Sequential()
model.add(LSTM(128, input_shape=(3, 1), return_sequences=True))
model.add(GRU(16, return_sequences=True, activation='relu'))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. COMPILE
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='loss',
    mode='auto',
    patience=100,
    restore_best_weights=True
)
model.fit(x, y, epochs=100, batch_size=30)

#4. PREDICT
loss = model.evaluate(x, y)
result = model.predict(x_predict)
print('loss: ', loss, 'result: ', result)