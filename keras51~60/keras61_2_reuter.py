from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd


#1. DATA
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.1
)
print(x_train) 
print(y_train)                              # [ 3  4  3 ... 25  3 25]
print(x_train.shape, y_train.shape)         # (8982,) (8982,)
print(x_test.shape, y_test.shape)           # (2246,) (2246,)

# x DATA 길이 맞추기
print(len(x_train[0]), len(x_train[1]))     # 87 56
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(type(x_train), type(x_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print('뉴스기사의 최대 길이: ', max(len(i) for i in x_train)) # 뉴스기사의 최대 길이:  2376
print('뉴스기사의 평균 길이: ', sum(map(len, x_train)) / len(x_train)) # 뉴스기사의 평균 길이:  145.84948045522017

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre',
                        maxlen=100,
                        truncating='pre'
                        )
print(x_train.shape) # (10105, 100)
# 나머지 전처리
# 모델 구성
# 시작

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM
model = Sequential()
model.add(Embedding(46, 100)) # 됩니다..
model.add(LSTM(32))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Dense(1, activation='softmax'))
model.summary()

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=200)

#4. PREDICT
acc = model.evaluate(x_test, y_test)[1]
y_predict = model.predict(x_test)
print('acc: ', acc)
# acc:  0.04829292371869087









