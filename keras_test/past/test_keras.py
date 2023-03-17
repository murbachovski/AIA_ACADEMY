#1. DATA
#2. MODEL
#3. COMPILE
#4. EVALUATE, PREDICT
#5. RESULT

#1. DATA
import numpy as np
import tensorflow as tf    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
x = np.array(
    [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],  
    [7, 1.4],
    [8, 1.5],  
    [9, 1.6],
    [10, 1.4]] # 10행2열
)
y = np.array(
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
)
print(x.shape)
print(y.shape)

#2. MODEL
model = Sequential()
model.add(Dense(3, input_dim = 2)) #first hidden layout
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 10)

#4. EVALUATE
loss = model.evaluate(x, y)
print("loss:", loss)

#5. RESULT
result = model.predict([[10, 1.4]]) #열우선, 데이터 구조 맞추어 주기.
#23.03.02 다른 데이터 구조가 들어왔을때 어떻게 맞추는가?
print("[[10, 14]]의 예측 값: ", result)
