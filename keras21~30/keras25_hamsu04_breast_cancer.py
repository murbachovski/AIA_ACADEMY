import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

#1. DATA
dataset = load_breast_cancer()
# print(dataset)  # PANDS : .describe()
# print(dataset.DESCR) # PANDSE : .columns()
# print(dataset.feature_names)
x = dataset.data # == dataset['data']
y = dataset.target
# print(x.shape)  (569, 30)
# print(y.shape)  (569,) [한개]    ==> (569, 1) == [[두개]]
# print(y)
print(x.shape, y.shape)
#2. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    test_size=0.2
)

#3. MODEL
# model = Sequential()
# model.add(Dense(10, input_dim = x.shape[1], activation='relu')) # input_dim = 30
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(30,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

# input1 = Input(shape=(10,))
# dense1 = Dense(256)(input1)
# dense2 = Dense(128)(dense1)
# dense3 = Dense(64)(dense2)
# dense4 = Dense(32)(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)

#4. COMPILE
model.compile(loss = 'binary_crossentropy', # loss: [0.2517603039741516, 0.9385964870452881, 0.05076000839471817]
              optimizer='adam',                     #'binary_crossentropy'      'accuracy'           'mse'
              metrics=['accuracy'] # 'acc', 'mse', 'mean_squared_error'
              )             # loss = 'mse' ==> 실수값  
hist = model.fit(x_train,          # loss = 'binary_crossentropy' ==> 
          y_train,
          epochs=20,
          batch_size=10,
          validation_split=0.2,
          verbose=1,
          )

#5. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results:', results)
y_predict = np.round(model.predict(x_test))
# print("############################################")
# print(y_test[:5])
# print(y_predict[:5])
# # print(np.round(y_predict[:5]))
# print("############################################")
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score # MSE
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)
print(y_test)
#acc:  0.9385964912280702
import matplotlib.pyplot as plt  
plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['accuracy'], marker = '.', c='red', label ='acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c='blue', label ='val_accuracy')
# print(hist.history)
plt.title('load_breast_cancer')
plt.xlabel('epochs')
plt.ylabel('accuracy, val_accuracy')
plt.legend()
plt.grid()
plt.show()