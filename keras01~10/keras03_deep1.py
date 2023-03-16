# 1.data
import numpy as np # 수치화
x = np.array([1, 2, 3]) # /
y = np.array([1, 2, 3]) # / #행렬 연산
# node = 신경망 각각 하나


# model
import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적인 신경망 모델(Sequential)
from tensorflow.keras.layers import Dense # 신경망의 층

model = Sequential()
model.add(Dense(3, input_dim=1)) # 신경망 층 구현
model.add(Dense(4, input_dim=3)) # 2층 신경망 # input_dim은 1층(전 단계)으로 적용되니 명시해주지 않아도 됨. 
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1)) # [1, 2, 3]통으로 들어간 것
    #                                  0        input layer
    #                                0 0 0
    #                               0 0 0 0
    #                              0 0 0 0 0    hidden layer
    #                                0 0 0
    #                                  0        output layer

#3 컴파일(compile), 훈련
model.compile(loss = 'mse', optimizer ='adam')  
model.fit(x, y, epochs=100)

# loss: 0.0018 기본 설정이 제일 잘 나오네요.

# 역전파
