import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1. DATA
path = ('./_data/dacon_diabetes/')
path_save = ('./_save/dacon_diabetes/')
#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL
train_csv = train_csv.dropna()
print(train_csv.shape)
print(train_csv.columns)
print(test_csv.shape)
print(test_csv.columns)
#1-5. DROP(x, y DATA SPLIT)
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

#2. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=9999,
    test_size=0.05
)

#3. MODEL
model = Sequential()
model.add(Dense(9, input_dim = x.shape[1], activation='relu')) 
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

#4. COMPILE
model.compile(loss = 'binary_crossentropy', 
              optimizer='adam',                   
              metrics=['accuracy'] 
              )    
es = EarlyStopping(monitor='val_accuracy',
                   patience=2000,
                   mode='max',
                   verbose=1,
                   restore_best_weights=True
                   )
hist = model.fit(x_train,        
          y_train,
          epochs=10000,
          batch_size=13,
          validation_split=0.4,
          verbose=1,
          callbacks=[es]
          )

model.save('./_save/keras26_dacon_daiabetes_save_model.h5')

#5. EVALUATE, PREDICT
results = model.evaluate(x_test, y_test)
print('results:', results)
y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score # MSE
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

#6. SUBMISSION_CSV
y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'diabetes_new_last_submit.csv')

import matplotlib.pyplot as plt
plt.plot(hist.history['val_accuracy'])
plt.show()