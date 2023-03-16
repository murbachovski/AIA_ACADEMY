from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(type(x))
# print(x)

# #정규화 변환
# print(np.min(x), np.max(x))     #0.0 711.0
# MinMaxScaler() loss:  603.4655151367188
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.min(x), np.max(x))     #0.0 1.0
# StandardScaler()
# scaler = StandardScaler() # loss:  590.4216918945312
# scaler.fit(x)
# x = scaler.transform(x)
# MaxAbsScaler() # loss:  572.6917724609375
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# scaler = RobustScaler() #loss:  559.976806640625
# scaler.fit(x)
# x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))     #-0.005578376185404939 1.1478180091225065

#2. MODEL
input1 = Input(shape = (13,), name='h1')
dense1 = Dense(30, name='h2')(input1)
dense2 = Dense(20, name='h3')(dense1)
dense3 = Dense(10, name='h4')(dense2)
output1 = Dense(1, name='h5')(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()

# # 데이터가 3차원이면(시계열 데이터)
# # (1000, 100, 1) ==> input_shape=(100, 1)

# # 데이터가 4차원이면(이미지 데이터)
# # (60000, 32, 32, 3) ==> input_shape=(32, 32, 3)
 


# #3. COMPILE
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=10)

# #4. PREDICT
# loss = model.evaluate(x_test, y_test)
# print('loss: ', loss)