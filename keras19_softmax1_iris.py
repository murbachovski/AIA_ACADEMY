import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import  accuracy_score # 분류하면? accuracy_score떠올라야해

#1. DATA
dataset = load_iris()
print(dataset.DESCR)            #판다스 describe()
print(dataset.feature_names)    #판다스 columns

x = dataset.data
y = dataset['target']
print(x.shape, y.shape) #(150, 4) (150,)
print(x)
print(y)
print('y.label(라벨 값): ', np.unique(y)) #y.label(라벨 값):  [0 1 2]

###############################요지점에서 원핫을 해야한다.################################
#y를 (150, ) => (150, 3)
from keras.utils import to_categorical
#판다스에 겟더미
#사이킷런에 원핫인코더
y = to_categorical(y)
print(y)
print(y.shape)
#########################################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    # random_state=333,
    train_size=0.9,
    stratify=y  # 일정한 비율로 분배해준다.
)
print(y_train)
print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2]), array([5, 5, 5], dtype=int64))

#2. MODEL
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=x.shape[1]))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax')) #y.shape[1]될까? softmax y_라벨의 갯수만큼 output_layer에 적용.

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(mode='auto',
                   patience=20,
                   monitor='val_acc',
                   restore_best_weights=True,
                   verbose=1
                   )
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])

#4. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results: ', results)
y_predict = np.round(model.predict(x_test))
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score # MSE
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

import matplotlib.pyplot as plt  
plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['acc'], marker = '.', c='red', label ='acc')
plt.plot(hist.history['val_acc'], marker = '.', c='blue', label ='val_acc')
# print(hist.history)
plt.title('load_breast_cancer')
plt.xlabel('epochs')
plt.ylabel('acc, val_acc')
plt.legend()
plt.grid()
plt.show()

# + 얼리스타핑
# + 에큐러시스코어

#accuracy_score를 사용해서 스코어를 빼세요.