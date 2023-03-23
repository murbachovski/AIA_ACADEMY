from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, GRU, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# [실습] 맹그러
# 목표는 cnn성능보다 좋게 맹그러!

#1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)

#reshape
x_train = x_train.reshape(-1, 32, 96)
x_test = x_test.reshape(-1, 32, 96)

#2. MODEL
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(32, 96))) # == model.add(Dense(64, input_shape=(28*28,)))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='softmax'))


#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.025)

#4. EVALUATE
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])

#LSTM
# loss:  0.0 acc:  0.10000000149011612

#Conv1D
# loss:  0.0 acc:  0.10000000149011612