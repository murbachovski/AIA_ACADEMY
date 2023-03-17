#Alone TEST
#1. DATA
#2. MODEL
#3. COMPILE
#4. EVALUATE, PREDICT

#1. DATA
import numpy as np
x = np.array([range(10)]) # (1, 10)
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (1, 5)
new_x = x.T # (10, 1)
new_y = y.T # (5, 1)

#2. MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(5)) 
model.add(Dense(4)) 
model.add(Dense(1))

#3. COMPILE
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    new_x,
    new_y,
    random_state = 123
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 3)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
result = model.predict([10])
print("result:", result)