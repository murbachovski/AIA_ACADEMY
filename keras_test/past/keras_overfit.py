import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # MSE
from tensorflow.python.keras.callbacks import EarlyStopping
#KAGGLE_BIKE

#1. DATA
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')
#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL
train_csv = train_csv.dropna()

#1-5. DROP(x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=8
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = x.shape[1]))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=10,
    verbose=1
    )
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#5. DEF
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

#6. SUBMISSION
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sampleSubmission.csv')
submission['count'] = y_submit
submission.to_csv(path_save + 'TEST_SUBMIT.csv')
