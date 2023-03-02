import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   

#1. DATA
x = np.array([range(10), range(21, 31), range(201, 211)])

print(x)    # 3, 10
x = x.T     # 10, 3
print(x.shape)  

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) #1행 10열
y = y.T #10, 1

# 2. model
model = Sequential()
model.add(Dense(3, input_dim = 3)) # Dense = 내 마음대로 // input_dim = 열, column, 차원의 갯수
model.add(Dense(5)) 
model.add(Dense(4)) 
model.add(Dense(1)) 

# 3. Compile, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 30, batch_size = 3)

#4. evaluate
loss = model.evaluate(x, y)
print("loss:", loss)

#5. result
result = model.predict([[9, 30, 210]]) #열우선. 데이터 구조 맞추어 주기. 
print("[[9, 30, 210]]의 예측값:", result)