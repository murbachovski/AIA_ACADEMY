from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. DATA
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 실습 : 잘라봐
#trian_test_split로만 잘라라. # 10:3:3
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.625, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.5, random_state=1)

# x_val = x_train[14:]
# y_val = y_train[14:]

# x_test = x_train[11:14]
# y_test = y_train[11:14]
# is it right?

#모의고사 == train == 검증
# x_val = np.array([14, 15, 16])
# y_val = np.array([14, 15, 16])

# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])


#2. MODEL
model = Sequential()
model.add(Dense(256, activation='relu', input_dim = 1))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. COMPILE
model.compile(loss= 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1,
          validation_data=(x_val, y_val)
          )

#4. VALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
result = model.predict([17])
print('17예측은?: ', result)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_test, color='red')
plt.show()