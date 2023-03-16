import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score #이곳에 accuracy_score가 들어 있다.
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical #ONE_HOT_EDCODING
import matplotlib.pyplot as plt  #시각화
from sklearn.datasets import load_wine
import sklearn

#1. DATA
dataset = load_wine()
x = dataset.data
y = dataset['target']
# print(x.shape, y.shape) #(178, 13) (178,)
# print('y.label(라벨 값): ', np.unique(y)) #y.label(라벨 값):  [0 1 2]

#2. ONE_HOT_ENCODING
# y = to_categorical(y)
# print(y)
# print(y.shape) #(178, 3)

# 2-1. ONE_HOT_ENCODING_PANDAS
y = dataset['target']
y = pd.get_dummies(y)
# print(y)
# print(b.shape) #(178, 3)

#2-2. ONE_HOT_ENCODING_SKLEARN
# a = dataset['target']
# label_binarizer = sklearn.preprocessing.LabelBinarizer()
# label_binarizer.fit(range(max(a)+1))
# b = label_binarizer.transform(a)
# print(b)
# print(b.shape) #(178, 3)
# from sklearn.preprocessing import OneHotEncoder
# OneHotEncoder(
#     categories='auto',  # Categories per feature
#     drop=None, # Whether to drop one of the features
#     sparse=True, # Will return sparse matrix if set True
# )
# df = load_wine()
# x = df.data
# y = df.target
# ohe = OneHotEncoder()
# transformed = ohe.fit_transform('pipeline')
# print(transformed.toarray())

# df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Green']})
# one_hot = pd.get_dummies(df['Color'])
# df = df.join(one_hot)
# print(df)
# print(df.shape)


#3.TRAIN_TEST_SPLIT <stratify=y가 들어갑니다  # 일정한 비율로 분배해준다.>
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    stratify=y
)
print(y_train)
print(np.unique(y_train, return_counts=True)) #(array([0., 1.], dtype=float32), array([284, 142], dtype=int64))

#4. MODEL

model = Sequential()
model.add(Dense(32, input_dim=x.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

#5. COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) #acc들어가는 이유는?

#6. EARLYSTOPPINGs
es = EarlyStopping(
    monitor='val_loss', #이곳에 변경을 잘해야 된다.
    patience=50,
    mode='min',
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, callbacks=[es], validation_split=0.2, batch_size=5, verbose=1)

#7. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results: ', results)
y_predict =  np.round(model.predict(x_test)) 
acc = accuracy_score(y_test, y_predict)     #r2 = r2_score(y_test, y_predict) r2_score와 같은 구조네?
print('acc: ', acc)
