from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, LSTM, Conv1D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score        
from tensorflow.python.keras.callbacks import EarlyStopping                                 

#1. DATA
dataset = load_wine()
x = dataset.data
y = dataset.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=333
)
print(x_train.shape, x_test.shape) #(142, 13) (36, 13)

scaler = RobustScaler()   
scaler.fit(x_train)      
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)

print(np.unique(y, return_counts=True))

#2. MODEL
model = Sequential()
model.add(Conv1D(256, 2, input_shape=(13, 1)))
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

#3. COMPILE
es = EarlyStopping(monitor='acc', patience = 20, mode='max'
              ,verbose=1
              ,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])  #다중분류에서 loss는 
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, verbose=1)

#4. PREDICT
results = model.evaluate(x_test, y_test)
print(results)
print('loss:', results[0])
print('acc:',results[1])

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)  
y_pred = np.argmax(y_predict, axis = 1) 
acc = accuracy_score(y_test_acc, y_pred) 

print('acc:', acc)

# [0.32748353481292725, 0.9444444179534912]
# loss: 0.32748353481292725
# acc: 0.9444444179534912
# acc: 0.9444444444444444

#Conv1D
# loss: 0.12292537093162537
# acc: 0.9444444179534912
# acc: 0.9444444444444444