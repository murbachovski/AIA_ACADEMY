#1. DATA
import numpy as np

x1_datasets = np.array([range(100), range(301,401)]) #삼성, 아모레
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) #온도, 습도, 강수량
# print(x1_datasets.shape)        # (2, 100)
# print(x2_datasets.shape)        # (3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
# print(x1.shape)                 # (100, 2)
# print(x2.shape)                 # (100, 3)

y = np.array(range(2001,2101)) #환율

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,
    x2,
    y,
    train_size=0.7,
    random_state=333
)

# print(x1_train.shape, x1_test.shape)
# print(x2_train.shape, x2_test.shape)
# print(y_train.shape, y_test.shape)
# (70, 2) (30, 2)
# (70, 3) (30, 3)
# (70,) (30,)

#2. MODEL
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input


#2-1 MODEL1
input1 = Input(shape=(2,))
Dense1 = Dense(10, activation='relu', name='stock1')(input1)
Dense2 = Dense(10, activation='relu', name='stock2')(Dense1)
Dense3 = Dense(10, activation='relu', name='stock3')(Dense2)
output1 = Dense(10, activation='relu', name='output1')(Dense3)

#2-2. MODEL2
input2 = Input(shape=(3,))
Dense11 = Dense(10, name='weather1')(input2)
Dense12 = Dense(10, name='weather2')(Dense11)
Dense13 = Dense(10, name='weather3')(Dense12)
Dense14 = Dense(10, name='weather4')(Dense13)
output2 = Dense(10, name='output2')(Dense14)

from tensorflow.keras.layers import concatenate, Concatenate # (소문자)함수, (대문자)클래스
merge1 = concatenate([output1, output2], name='mg1') #리스트 형태로 받아들임.
merge2 = Dense(1, activation='relu', name='mg2')(merge1)
merge3 = Dense(1, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
x_train = [x1_train, x2_train]
model.fit(x_train, y_train, epochs=10, batch_size=10)

#4. PREDICT
x_test = [x1_test, x2_test]
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
