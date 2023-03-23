# from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM, GRU, Conv1D
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd 
from tensorflow.keras.utils import to_categorical
#1. 데이터

path = './_data/kaggle_diabetes/'
path_save = './_save/kaggle_diabetes/'

datasets = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)
submission = pd.read_csv(path +'sample_submission.csv', index_col=0)

# print(train_csv.shape) #(10886, 11)
x = datasets.drop(['Outcome'], axis=1)
y = datasets['Outcome']
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape)  



scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
# print(x_train.shape, x_test.shape)  
x_train = x_train.reshape(-1, 4, 2)
x_test = x_test.reshape(-1, 4, 2)



#2. 모델 구성
model = Sequential()
model.add(Conv1D(256, 2, input_shape=(4, 2)))
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
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
start_time=time.time()
model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit(x_train, y_train,
                 epochs =10,
                 batch_size=8,
                 validation_split = 0.2,
                 callbacks=[es])

end_time=time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)


# result : 0.5758771300315857
# acc : 0.7099236641221374

# result : 0.617805540561676
# acc : 0.3511450381679389


#Conv1D
# result : 0.6029714345932007
# acc : 0.3511450381679389