from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
import numpy as np

# [실습] 맹그러
# 목표는 cnn성능보다 좋게 맹그러!

#1. DATA
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)
print(x_test.shape)

#SCALER
# scaler = RobustScaler()   
# scaler.fit(x_train)      
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#ONE_HOT
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#reshape
x_train = x_train.reshape(-1, 32, 96)
x_test = x_test.reshape(-1, 32, 96)

#2. MODEL
model = Sequential()
model.add(Conv1D(256, 3, input_shape=(32, 96))) # == model.add(Dense(64, input_shape=(28*28,)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax'))


#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.025, callbacks=[es])


#4. EVALUATE
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
print('loss: ', results[0], 'acc: ', acc)

#loss:  4.605197429656982 acc:  0.01

#Conv1D
# loss:  4.605199813842773 acc:  0.01