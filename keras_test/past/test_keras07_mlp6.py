import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([range(5), range(21, 26), range(101, 106)])
print(x)
print(x.shape)

new_x = x.T
print(new_x.shape)

y = np.array([
    [1, 2, 3, 4, 5],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [10, 20, 30, 40, 50]
])
print(x)
print(x.shape)
new_y = y.T
print(new_y.shape)

#2. MODEL
model = Sequential()
model.add(Dense(2, input_sdim = 3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(3)) #outputData구조 잘 맞추기. 행무시/열우선
#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(new_x, new_y, epochs = 10)
#4. EVAlUATE
loss = model.evaluate(new_x, new_y)
print("loss: ", loss)
#5. RESULT
result = model.predict([[4, 25, 105]])
print("result[4, 25, 105]", result)

#########new_x and new_y 데이터 구조 잘 맞추어 주기.#########