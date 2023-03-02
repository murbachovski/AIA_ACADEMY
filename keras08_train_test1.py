
import numpy as np  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

#1. DATA
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# 마지막에 온 ,는 딱히 상관없다.
# print(x)
# print(y)

x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

#2. MODEL
model = Sequential()
model.add(Dense(10, input_dim = 1)) 
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2)

#4. EVALUATE
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

#5. RESULT
result = model.predict([11])
print("[11]predict:", result)

 