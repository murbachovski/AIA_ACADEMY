import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x_data = [[0,0], [0,1], [1,0], [1,1]] # 4, 2
y_data = [0, 1, 1, 0] # 같으면 0 다르면 1

#2. MODEL
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. PREDICT
y_predict = model.predict(x_data)
results = model.evaluate(x_data, y_data)
print('model.score: ', results[1])


acc = accuracy_score(y_data, np.round(y_predict))
print('acc_score:', acc)

# model.score:  0.75
# acc_score: 0.75