import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # MSE
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
#1. DATA
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)   #?
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=32
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = x.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. COMPILE
model.compile(loss ='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
    verbose=1
    )
hist = model.fit(x_train, y_train, epochs = 1000, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es])

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#5 DEF
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("rmse: ", rmse)

#6. SUBMISSION_CSV
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'NEW_SUBMIT.csv')

#7. PLT
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker = '.', c ='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.',
         c='blue', label = 'val_loss'
         )
plt.title('KAGGLE_BIKE')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()

#1.
# loss:  23373.294921875
# r2:  0.30685244601303263
# rmse:  10.474323845637826

#2.
# loss:  23351.521484375
# r2:  0.30749827482221104
# rmse:  10.549256049266427

#3.
# loss:  23480.33984375
# r2:  0.30367807703358674
# rmse:  10.471255632280394