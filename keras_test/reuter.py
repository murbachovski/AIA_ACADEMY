from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

#1. DATA
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.1
)

print("뉴스기사의 최대길이: ", max(len(i) for i in x_train))
print("뉴스기사의 평균길이: ", sum(map(len, x_train)) / len(x_train))

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

x_train = x_train.reshape(-1, 100, 1)
x_test = x_test.reshape(-1, 100, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Flatten
model = Sequential()
model.add(Embedding(10000, 40, input_length=100))
model.add(LSTM(64))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(46, activation='softmax'))

#3. COMPIEL
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=8)

#4. PREDICT
acc = model.evaluate(x_test, y_test)[1]
print(acc)