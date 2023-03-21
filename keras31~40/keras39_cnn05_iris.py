from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D
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

scaler = RobustScaler()   
scaler.fit(x_train)      
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)

#2. MODEL
input1 = Input(shape=(2, 2, 1))
conv1 = Conv2D(16,(2,2), padding = 'same')(input1)
conv2 = Conv2D(32,2,padding='same')(conv1)
conv3 = Conv2D(16,2,padding='same')(conv2)
flat = Flatten()(conv3)
dense1 = Dense(16)(flat)
dense2 = Dense(8)(dense1)
dense3 = Dense(16, activation ='relu')(dense2)
drop1 = Dropout(0.3)(dense3)
output1 = Dense(3, activation='softmax')(drop1)
model = Model(inputs= input1, outputs=output1)

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