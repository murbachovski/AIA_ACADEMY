#1.DATA
#2.MODEL
#3.COMPILE
#4.EVALUATE, PREDICT

import numpy as np                        
import tensorflow as tf       
from tensorflow.keras.models import Sequential     
from tensorflow.keras.layers import Dense    
from sklearn.datasets import load_digits
#1. DATA
dataset = load_digits()

x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (1797, 64) (1797, )

#2. MODEL
model = Sequential()
model.add(Dense(516, input_dim = 64))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. COMPILE
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=10
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 10)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2_SCORE:", r2)