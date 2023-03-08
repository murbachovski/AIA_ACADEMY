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
model.add(Dense(10, input_dim = x.shape[1]))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True) #통상적으로 loss보다 val_loss가 낫다. # auto = accuracy/loss/val_loss 자동으로 mode설정해줍니다.
hist = model.fit(x_train, y_train, epochs = 2000, batch_size = 16, validation_split=0.2, verbose = 1, callbacks=[es])

#4. EVALUATE, PREDICT
from sklearn.metrics import r2_score, mean_absolute_error # MSE
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#그림 그리기
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
