from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
import numpy as np

# [실습] 맹그러
# 목표는 cnn성능보다 좋게 맹그러!

#1. DATA
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)



print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784) 
print(y_train.shape, y_test.shape) #(60000,) (10000,)

#SCALER
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)
print(y_train.shape, y_test.shape) #(60000,) (10000,)
print(y_train, y_test)

#ONE_HOT
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train, y_test)
print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

#reshape
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

#2. MODEL
model = Sequential()
model.add(LSTM(256, input_shape=(28, 28))) # == model.add(Dense(64, input_shape=(28*28,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))


#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.025, callbacks=[es])

#4. EVALUATE
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
print('loss: ', results[0], 'acc: ', results[1], 'acc: ', acc)

# loss:  0.3894904851913452 acc:  0.8604999780654907 acc:  0.8605