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
model.add(Dense(256, activation='relu', input_dim = 13))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 50, validation_split=0.2, verbose = 1)
print(hist.history)
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'], color = 'red')
plt.show()
