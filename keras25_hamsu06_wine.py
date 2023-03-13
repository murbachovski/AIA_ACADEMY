from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score                                         

#1. DATA
dataset = load_wine()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
# print(type(x))
# print(x)

# #정규화 변환
# print(np.min(x), np.max(x))
# MinMaxScaler()                # acc:  0.3055555555555556
scaler = MinMaxScaler() 
scaler.fit(x)
x = scaler.transform(x)
# StandardScaler()              # acc:  0.19444444444444445
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# MaxAbsScaler()                # acc:  0.3055555555555556
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# RobustScaler()                # acc:  0.16666666666666666
# scaler = RobustScaler()         
# scaler.fit(x)
# x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2. MODEL
# model = Sequential()
# model.add(Dense(1, input_dim=x.shape[1]))
# model.add(Dense(1, input_dim = 'softmax'))

input1 = Input(shape=(13,))
dense1 = Dense(1)(input1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)



#3. COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
# print('loss: ', loss)
y_predict = model.predict(x_test)
y_predict = np.around(y_predict)
t_test_acc = np.around(y_test)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)