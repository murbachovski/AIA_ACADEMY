from sklearn.datasets import load_diabetes

#1. DATA
dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (442, 10) (442,) input_dim = 10 

#[실습]
# R2 = 0.62 이상

#2. MODEL
import numpy as np
import tensorflow as tf     
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    

model = Sequential()
model.add(Dense(40, input_dim = 10))
model.add(Dense(32))
model.add(Dense(53))
model.add(Dense(53))
model.add(Dense(53))
model.add(Dense(53))
model.add(Dense(53))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))


#3. COMPILE
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.9,
    random_state=123
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 100)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # GOOD, Y = wX + b, 보조지표
print("R2_SCORE: ", r2)

#r2_SCORE = 0.51
#R2_SCORE:  0.6432587748923371