import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score # accuracy_score기억해!
from tensorflow.python.keras.callbacks import EarlyStopping # .callback import EarlyStopping기억해!
from sklearn.datasets import load_digits


#1. DATA
dataset = load_digits()

x = dataset.data    #dataset['data']
y = dataset.target  #dataset['target]
print('y.label: ', np.unique(y))

#1-2. ONE_HOT_ENCODING
from keras.utils import to_categorical
y = to_categorical(y)

#1-2. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.8,
    random_state=32,
    stratify=y      #TRAIN_TEST_SPLIT에서 stratify=y를 넣어줘야해!
                    #일정한 비율로 분배해줍니다.  
)
print(np.unique(y_train, return_counts=True))

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim=x.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

#3. COMPILE #여기서 EarlyStopping해준다. 기억해라!
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) #metrics=['acc']기억해라!
es = EarlyStopping(
    mode='auto',
    patience=10,
    monitor='val_acc',
    restore_best_weights=True,
    verbose=1
)
his = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[es], verbose=1)

#4. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print(results)
y_predict = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

#5. SIGHT
# import matplotlib.pyplot as plt
# plt.plot()
# plt.plot()
# plt.title()
# plt.xlabel()
# plt.ylabel()
# plt.legend()
# plt.grid()
# plt.show()