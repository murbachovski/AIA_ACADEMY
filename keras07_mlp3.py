import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  

#1. data
x = np.array(
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# PREDICT [[10, 1.4, 0]]

trans_x = x.transpose()
print(x)
#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim = 3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(trans_x, y, epochs = 100)

#4. 평가, 예측
loss = model.evaluate(trans_x, y)
print("loss :", loss)

#5. RESULT
result = model.predict([[10, 1.4, 0]])
print("result :", result)

#loss 7.3
#loss 8.9
#loss 3.3
#loss 2.9
#loss 2.1