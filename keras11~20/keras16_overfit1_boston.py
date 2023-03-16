from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset['target']
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=123,
    test_size=0.2
)

#2. MODEL
model = Sequential()
model.add(Dense(5, input_dim = x.shape[1]))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 8, validation_split=0.2, verbose = 1)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(hist)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(hist.history)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(hist.history['loss'])
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c='red', label ='로쓰')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label ='발로쓰')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
