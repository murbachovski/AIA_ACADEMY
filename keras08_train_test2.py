import numpy as np  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

#1. DATA
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

#[실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[:7]  # [1 2 3 4 5 6 7]
print(x_train)
x_test = x[7:]   # [ 8  9 10]
print(x_test)
y_train = y[:7]
print(y_train)
y_test = y[7:]
print(y_test)
print(x_train.shape, x_test.shape) # (7, ) (3, )
print(y_train.shape, y_test.shape) # (7, ) (3, )

#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100)

#4. 평가
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

#5. PREDICT
result = model.predict([10])
print(result)