from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Reshape, Embedding, Conv1D
#1. DATA
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요.', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요', '나는 성호가 정말 재미없다 너무 정말'
        ]
print(docs[-1])

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0,0])

# 수치화까지 작업. 펼치기 전까지
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5, '최고에요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글세요': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밋네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}\
x = token.texts_to_sequences(docs)
y = labels
# print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

from tensorflow.keras.preprocessing.sequence import pad_sequences # 이 녀석이였구나
pad_x = pad_sequences(x, padding='pre', maxlen=5) # maxlen이 작으면 큰 놈들은 잘려 나간다. pre <-> post
# pad_x = pad_x.reshape(14, 5, 1)
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)
# print(pad_x)
# [[ 0  0  0  2  5]
#  [ 0  0  0  1  6]
#  [ 0  1  3  7  8]
#  [ 0  0  9 10 11]
#  [12 13 14 15 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0  0 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0 21 22]
#  [ 0  0  0  0 23]
#  [ 0  0  0  2 24]
#  [ 0  0  0  1 25]
#  [ 0  4  3 26 27]
#  [ 0  0  0  4 28]]
# print(pad_x.shape) # (14, 5)
word_size = len(token.word_index)
# print('단어 사전 갯수: ', word_size) # 단어 사전 갯수 : 28

#2. MODEL
model = Sequential()
model.add(Embedding(29, 32, input_length=5))
model.add(Conv1D(32, 2))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

#3. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=50, batch_size=8)

#4. PREDICT
acc = model.evaluate(pad_x, labels)[1]
print('acc: ', acc)
y_predict = model.predict(pad_x)[-1]
if y_predict < 0.5:
    y_predict = 0
print('나는 성호가 정말 재미없다 너무 정말 긍정1, 부정0: ', y_predict)
# # Dense_acc:  0.7857142686843872
# # LSTM_acc:  0.9285714030265808
# # Reshape_acc:  0.9285714030265808
# # Embdding_acc:  0.9285714030265808