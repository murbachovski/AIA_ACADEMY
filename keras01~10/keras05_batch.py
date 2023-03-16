import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  

# 1. Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

#2. model
model = Sequential()
model.add(Dense(3, input_dim=1)) # first hidden layout
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. compile
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 1)
# default batch size = 32