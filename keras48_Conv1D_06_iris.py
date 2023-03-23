from sklearn.datasets import load_iris
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
dataset = load_iris()
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
print(x_train.shape, x_test.shape) #(120, 4) (30, 4)
print(y_train.shape, y_test.shape) #(120, 3) (30, 3)

scaler = RobustScaler()   
scaler.fit(x_train)      
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

#2. MODEL
model = Sequential()
model.add(Conv1D(256, 2, input_shape=(4, 1)))
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
model.fit(x_train, y_train, epochs=100, batch_size = 1, verbose=1)

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
# loss: 0.09432816505432129
# acc: 0.9666666388511658
# acc: 0.9666666666666667

#Conv1D
# loss: 0.3815191388130188
# acc: 0.9333333373069763
# acc: 0.9333333333333333