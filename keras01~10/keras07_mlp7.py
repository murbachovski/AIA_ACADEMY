# x = 1개
# y = 3개

import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([range(10)])
print(x.shape)    # 
new_x = x.T       # 10, 3

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
              ])
new_y = y.T       # (10, 3)

# [실습]
# [예측]: [[9]] ==> y값 [[10, 1.9, 0]]

#2. MODEL
model = Sequential() # InputData
model.add(Dense(3, input_dim = 1)) #First Hidden Layout
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3)) #OutputData

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(new_x, new_y, epochs = 50)

#4. 평가, 예측
loss = model.evaluate(new_x, new_y)
# 훈련에 사용한 데이터는 평가에 사용하지 않는다.
# 1~7 훈련// 8~10평가
print("loss: ", loss)

#5. RESULT
result = model.predict([[9]]) # 보고 싶은 x데이터(열) 넣어주기.
print("result[[9]]일때 y값은 :", result)
