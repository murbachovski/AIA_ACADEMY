#1. DATA
import numpy as np
from sklearn.metrics import r2_score

x1_datasets = np.array([range(100), range(301,401)]) #삼성, 아모레
# x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) #온도, 습도, 강수량
# x3_datasets = np.array([range(201,301), range(511,611), range(1300,1400)]) #온도, 습도, 강수량
# print(x1_datasets.shape)        # (2, 100)
# print(x2_datasets.shape)        # (3, 100)

x1 = np.transpose(x1_datasets)
# x2 = x2_datasets.T
# x3 = x3_datasets.T
# print(x1.shape)                 # (100, 2)
# print(x2.shape)                 # (100, 3)

y1 = np.array(range(2001,2101)) #환율
y2 = np.array(range(1001,1101)) #금리

from sklearn.model_selection import train_test_split

x1_train, x1_test, y_train, y_test, y2_train, y2_test = train_test_split(
    x1,
    y1,
    y2,
    train_size=0.7,
    random_state=333
)
#random_state 꼭 맞춰주어야해
# x3_train, x3_test = train_test_split(
#     x3,
#     train_size=0.7,
#     random_state=333
# )
# y2_train, y2_test = train_test_split(
#     y2,
#     train_size=0.7,
#     random_state=333
#     )
# print(x1_train.shape, x1_test.shape)
# print(x2_train.shape, x2_test.shape)
# print(x3_train.shape, x3_test.shape)
# print(y_train.shape, y_test.shape)
# print(y2_train.shape, y2_test.shape)
# (70, 2) (30, 2)
# (70, 3) (30, 3)
# (70, 3) (30, 3)
# (70,) (30,)
# (70,) (30,)

#2. MODEL
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input


#2-1 MODEL1
input1 = Input(shape=(2,))
Dense1 = Dense(256, activation='relu', name='stock1')(input1)
Dense2 = Dense(128, activation='relu', name='stock2')(Dense1)
Dense3 = Dense(64, activation='relu', name='stock3')(Dense2)
output1 = Dense(32, activation='relu', name='output1')(Dense3)

# #2-2. MODEL2
# input2 = Input(shape=(3,))
# Dense11 = Dense(256, name='weather1', activation='relu')(input2)
# Dense12 = Dense(128, name='weather2', activation='relu')(Dense11)
# Dense13 = Dense(64, name='weather3')(Dense12)
# Dense14 = Dense(32, name='weather4')(Dense13)
# output2 = Dense(16, name='output2')(Dense14)

# #2-2. MODEL3
# input3 = Input(shape=(3,))
# Dense111 = Dense(256, name='feel1', activation='relu')(input2)
# Dense122 = Dense(128, name='feel2', activation='relu')(Dense111)
# Dense133 = Dense(64, name='fee3')(Dense122)
# Dense144 = Dense(32, name='feel4')(Dense133)
# output3 = Dense(16, name='output3')(Dense144)

#2-3 merge
from tensorflow.keras.layers import concatenate, Concatenate # (소문자)함수, (대문자)클래스
merge1 = concatenate([output1], name='mg1') #리스트 형태로 받아들임.
merge2 = Dense(32, activation='relu', name='mg2')(merge1)
merge3 = Dense(16, activation='relu', name='mg3')(merge2)
hidden_output = Dense(1, name='last')(merge3)

#2-4 분기1
bungi1 = Dense(10, activation='relu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

#2-5 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)


model = Model(inputs=[input1], outputs=[last_output1, last_output2])

model.summary()

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train], [y_train, y2_train], epochs=10)

#4. PREDICT
results = model.evaluate([x1_test], [y_test, y2_test])
print(results)
y_predict = model.predict([x1_test])
print(y_predict)
print(len(y_predict), len(y_predict[0]))   # 2, 30  List는 len으로 확인합니다. shape로 찍어서 보고 싶다면 np.array() 
r2_1 = r2_score(y_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[0])
print('r2_SCORE : ', (r2_1 + r2_2)/2)

# #3. COMPILE
# model1.compile(loss = 'mse', optimizer='adam')
# x_train = [x1_train, x2_train, x3_train]
# model1.fit(x_train, y_train, epochs=10, batch_size=10)

# #4. PREDICT
# x_test = [x1_test, x2_test, x3_test]
# loss1 = model1.evaluate(x_test, y_test)
# y_predict1 = model1.predict(x_test)
# r2_1 = r2_score(y_test, y_predict1)
# print('loss: ', loss1, 'r2: ', r2_1)

# #3. COMPILE
# model2.compile(loss = 'mse', optimizer='adam')
# x_train = [x1_train, x2_train, x3_train]
# model2.fit(x_train, y2_train, epochs=10, batch_size=10)

# #4. PREDICT
# x_test = [x1_test, x2_test, x3_test]
# loss2 = model2.evaluate(x_test, y2_test)
# y_predict2 = model2.predict(x_test)
# r2_2 = r2_score(y2_test, y_predict2)
# print('loss: ', loss2, 'r2_2: ', r2_2)

# loss:  14680.4580078125
# relu 적용 후 : loss:  15532.751953125

# y1 = loss:  14989.0126953125  r2:  -24.39726179775138
# y2 = loss:  2260.575927734375 r2_2:  -2.830301976541641

