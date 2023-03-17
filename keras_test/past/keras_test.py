from sklearn.datasets import load_linnerud
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
dataset = load_linnerud()
x = dataset.data
y = dataset.target
# print(x)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(y)
# print(x.shape)
# print(y.shape)

encoder = OneHotEncoder()
y = y.reshape(-1, 1)
y = encoder.fit_transform(y).toarray()

#1-1. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=32,
    stratify=y
)

#2. MODEL
model = Sequential()
model.add(Dense(32, input_dim=x.shape[1], activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(y.shape[1], activation='softmax'))

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10,
    restore_best_weights=True
)
model.fit(x_test, y_test, epochs=10, validation_split=0.2, batch_size=12)

#4. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results: ', results)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, y_predict)