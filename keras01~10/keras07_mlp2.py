import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  

#1. data
x = np.array(
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]
)

# (2, 10)을 (10, 2)로 바꾸세요.  전치행렬
# np.transpose(x)
# x.transpose()
# print(x.transpose().shape)
# x = x.T
# print("np.transpose(x) : ", np.transpose(x))
# print("np.transpose(x).shape :", np.transpose(x).shape)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

 
# print(x.shape) # (10, 2) -> 2개의 특성을 가진 10개에 데이터
# print(y.shape) # (10, )
# 열 = column 
# 행무시, 열우선
# 행렬 문제 10개 만들기. 목요일

'''
# 2. model
model = Sequential()
model.add(Dense(3, input_dim = 2)) # Dense = 내 마음대로 // input_dim = 열, column, 차원의 갯수
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
result = model.predict([[10, 1.4]]) #열우선. 데이터 구조 맞추어 주기. 
print("[[10, 14]]의 예측값:", result)

#23.03.02
# 스칼라 -> 벡터 -> 행렬 -> Tensor -> 4차원 Tensor
# 0 -> 1 -> 2 -> 3차원 -> 4차원 Tensor
# (10. 2) 2 = 열, 컬럼, 피쳐, 특성
'''
