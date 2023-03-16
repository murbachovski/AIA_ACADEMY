from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1. DATA
dataset = fetch_covtype()
x = dataset.data
y = dataset.target

# print(x.shape)
# print(y.shape)

#1. ONEHOTENCODING
# y = to_categorical(y)

#1. ONEHOEENCODING_SKLEARN
encoder = OneHotEncoder()
y = y.reshape(-1, 1)
y = encoder.fit_transform(y).toarray()

#1. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

#2. MODEL
model = Sequential()
model.add(Dense(50, input_dim = x.shape[1]))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=50,
    restore_best_weights=True,
    verbose=1
)
model.fit(x_train, y_train, epochs=10000, batch_size=500 ,validation_split=0.2, callbacks=[es])

#4. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results', results)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict)
print('acc', acc)