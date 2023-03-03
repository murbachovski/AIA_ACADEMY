from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split



#1. DATA
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    shuffle=True,
    random_state=50
)

#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 2)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test) # loss > r2 두가지가 엉켰을 경우
print("loss: ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # GOOD, Y = wX + b, 보조지표
print("R2_SCORE: ", r2)

#R2 = 0.77
