from sklearn.datasets import fetch_california_housing

#1. DATA
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (20640, 8) (20640, )

#############[실습]#################################
# R2 = 0.55 ~ 0.6 이상
####################################################

#2. MODEL
import numpy as np
import tensorflow as tf     
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    

model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(1))


#3. COMPILE
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    random_state=15
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 30)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # GOOD, Y = wX + b, 보조지표
print("R2_SCORE: ", r2)
