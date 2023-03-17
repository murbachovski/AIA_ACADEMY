from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
# - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica

print(x.shape, y.shape) # (150, 4) (150,)
# print(type(y))

#1-1. ONE_HOT
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


#1-2. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=333,
    stratify=y
)
scaler.fit(x_train)
x = train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))
print('y.label(라벨 값): ', np.unique(y))

#2. MODEL
model = Sequential()
model.add(Dense(7, input_dim=x.shape[1]))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    patience=50,
    mode='auto',
    restore_best_weights=True
)
model.fit(x_train, y_train, validation_split=0.3, epochs=1000, batch_size=5, callbacks=[es])

#4. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results: ', results)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict)
print('acc: ', acc)