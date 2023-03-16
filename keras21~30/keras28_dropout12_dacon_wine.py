import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical # ONE_HOT_ENCODIN
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder # SCALER
import matplotlib.pyplot as plt 
import datetime #시간을 저장해주는 놈


date = datetime.datetime.now()
# print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
# print(date) # 0314_1115
filepath = ('./_save/MCP/wine/')
filename = '{epoch:04d}-{val_acc:.4f}-{val_loss:.4f}.hdf5'


#1. DATA
path = ('./_data/wine/')
path_save = ('./_save/wine/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 
# le = LabelEncoder() # 정의 핏 트랜스폼
# le.fit(train_csv['type'])
# aaa = le.transform(train_csv['type'])
# train_csv['type'] = aaa
# test_csv['type'] = le.transform(test_csv['type'])

# print(train_csv.shape, test_csv.shape) #(5497, 13) (1000, 12)
# print(train_csv.columns)

# ISNULL
train_csv = train_csv.dropna()

# x, y SPLIT
x = train_csv.drop(['quality', 'type'], axis=1)
y = train_csv['quality']
test_csv = test_csv.drop(['type'], axis=1)
# print(x)
print(y)
print(np.unique(y, return_counts=True))
#(array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

# ONE_HOT
y = to_categorical(y)



# TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=9999,
    # stratify=y
)

# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape, x_test.shape) #(4397, 11) (1100, 11)

# 2. MODEL
# model = Sequential()
# model.add(Dense(20, input_shape=(12,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# #3. COMPILE
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_acc',
#     patience=100,
#     mode='auto',
#     restore_best_weights=True,
# )
# mcp = ModelCheckpoint( 
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath= "".join([filepath, 'dacon_wine', date, '_', filename])
# )
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=300, validation_split=0.2, callbacks=[es, mcp])

# model.save('./_save/wine/wine3_model.h5')
model = load_model('_save\wine\wine6_model.h5')
es = EarlyStopping(
    monitor='val_acc',
    patience=300,
    mode='auto',
    restore_best_weights=True,
)
mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath= "".join([filepath, 'dacon_wine', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=10, batch_size=500, validation_split=0.5, callbacks=[es, mcp])

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
submission['quality'] = y_submit
submission.to_csv(path_save + 'wine4_submit'+ date + '.csv')

#6. PLT
plt.plot(hist.history['val_acc'],label='val_acc', color='red')
plt.plot(hist.history['acc'],label='acc', color='blue')
plt.plot(hist.history['loss'],label='loss', color='green')
plt.plot(hist.history['val_loss'],label='val_loss', color='yellow')
plt.legend()
plt.show()