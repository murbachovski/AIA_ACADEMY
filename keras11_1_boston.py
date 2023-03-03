from sklearn.datasets import load_boston

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target

# print(x)
# print(y)

print(dataset)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# (1, 13) input_dim = 13

print(dataset.DESCR)

print(x.shape, y.shape) # (506, 13) (506, ) # Vactor = 1

#######[실습]#######
#1. train 0.7
#2. R2 0.8 이상
##################################################################################

#2. MODEL
import numpy as np
import tensorflow as tf     
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    

model = Sequential()
model.add(Dense(40, input_dim = 13))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(80))
model.add(Dense(1))


#3. COMPILE
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    random_state=101
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 400, batch_size = 2)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # GOOD, Y = wX + b, 보조지표
print("R2_SCORE: ", r2)

#r2