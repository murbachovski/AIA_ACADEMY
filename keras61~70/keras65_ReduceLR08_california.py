import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#1. DATA
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (20640, 8) (20640, )

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
)

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. COMPILE
# model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = 'mse', optimizer=optimizer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience='10', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1, validation_batch_size=0.2, callbacks=[es, rlr])


#4. PREDICT
results = model.evaluate(x_test, y_test)

print('ACC:', results)

# 1.21.5 num
# 2.9.3 ten