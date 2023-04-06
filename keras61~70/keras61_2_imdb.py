from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

# print(x_train)
# print(y_train) # [1 0 0 ... 0 1 0]
# print(x_train.shape, x_test.shape) # (25000,) (25000,)
# print(np.unique(y_train, return_counts=True)) # 판다스에서는 value_counts
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# print(pd.value_counts(y_train))
# 1    12500
# 0    12500
# dtype: int64

print('영화평의 최대 길이: ', max(len(i) for i in x_train)) # 뉴스기사의 최대 길이: 2494
print('영화평의 평균 길이: ', sum(map(len, x_train)) / len(x_train)) # 뉴스기사의 평균 길이:  238.71364

#[실습]
#[맹그러봐]
#Embedding, input_dim=10000, sigmoid,
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre',
                        maxlen=100,
                        truncating='pre'
                        )
x_test = pad_sequences(x_test, padding='pre',
                        maxlen=100,
                        truncating='pre'
                        )


#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM
model = Sequential()
model.add(Embedding(10000, 200)) # 됩니다..
model.add(LSTM(32))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=64)

#4. PREDICT
acc = model.evaluate(x_train, y_train)[1]
print('acc: ', acc)
acc:  1.0