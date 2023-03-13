from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score                                         

#1. DATA
dataset = fetch_covtype()
x = dataset.data
y = dataset.target

# #정규화 변환
# print(np.min(x), np.max(x))
# MinMaxScaler()                # acc:  0.025627565553385023
# scaler = MinMaxScaler() 
# scaler.fit(x)
# x = scaler.transform(x)
# StandardScaler()              # acc:  0.13367124773026515
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# MaxAbsScaler()                # acc:  0.03195270345860262
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# RobustScaler()                # acc:  0.01699611886095884
scaler = RobustScaler()         
scaler.fit(x)
x = scaler.transform(x)

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
model = Sequential()
model.add(Dense(1, input_dim=x.shape[1]))
model.add(Dense(1, input_dim = 'softmax'))

#3. COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=10000)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
# print('loss: ', loss)
y_predict = model.predict(x_test)
y_predict = np.around(y_predict)
t_test_acc = np.around(y_test)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)