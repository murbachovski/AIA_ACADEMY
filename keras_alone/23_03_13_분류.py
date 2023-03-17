import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler, OneHotEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import datetime #시간을 저장해주는 놈
date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/dacon_diabetes/')
path_save = ('./_save/dacon_diabetes/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape)    # (652, 9) (116, 8)
# print(train_csv.feature_names) # AttributeError: 'DataFrame' object has no attribute 'feature_names'

#1-1. ISNULL
train_csv = train_csv.dropna()
# print(train_csv.shape) # (652, 9)

#1-2 x, y SPLIT
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

#1-2 ONE_HOT_ENCODING(TO_CATEGORICAL)
y = to_categorical(y)


#1-3. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.05,
    random_state=4444,
    # stratify=y
)

# 1-4. SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (521, 8) (131, 8)

# #2. Model
# model = Sequential()
# model.add(Dense(8, input_dim=x.shape[1]))
# model.add(Dropout(0.2))
# model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2, activation='softmax'))

# # #3. COMPILE
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_acc',
#     patience=200,
#     mode='auto',
#     restore_best_weights=True   
# )
# mcp = ModelCheckpoint( #restore_best_weights=True가 들어가 있는 모델
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath= "".join([filepath, 'k27_', date, '_', filename])
# )
# # print(x_train, y_train, x_train.shape, y_train.shape) # (521, 8) (521, 2)
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])

# # SAVE
# model.save('./_save/dacon_diabetes/당뇨병최종모델.h5')
# # LOAD
model = load_model('_save\dacon_diabetes\당뇨병최종모델.h5')
es = EarlyStopping(
    monitor='val_acc',
    patience=400,
    mode='auto',
    restore_best_weights=True   
)
mcp = ModelCheckpoint( #restore_best_weights=True가 들어가 있는 모델
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath= "".join([filepath, 'k27_', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=400, validation_split=0.1, callbacks=[es])

#4. PREDICT
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])
y_predict = model.predict(x_test)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc: ', acc)

#5. SUBMISSION_CSV
test_csv_sc = scaler.transform(test_csv)
y_submit = model.predict(test_csv_sc)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'all_submit4'+ date + '.csv')


#6. PLT
plt.plot(hist.history['val_acc'],label='val_acc', color='red')
plt.plot(hist.history['acc'],label='acc', color='blue')
plt.plot(hist.history['loss'],label='loss', color='green')
plt.plot(hist.history['val_loss'],label='val_loss', color='yellow')
plt.legend()
plt.show()

# 1. loss:  0.5424655675888062 acc:  0.7633587718009949
# 2. loss:  0.5916731953620911 acc:  0.732824444770813
# 3. loss:  0.5028010606765747 acc:  0.7404580116271973