#불러와서 모델 완성
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
path = 'd:/study_data/_data/cat_dog/Petimages/'
save_path = 'd:/study_data/_save/cat_dog/'

x_train = np.load(save_path + 'cd_x_train.npy')
x_test = np.load(save_path + 'cd_x_test.npy')
y_train = np.load(save_path + 'cd_y_train.npy')
y_test = np.load(save_path + 'cd_y_test.npy')


#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(200, 200, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=20,
    mode='max',
    restore_best_weights=True
)
hist = model.fit(x_train,
                 y_train,
                epochs=50,
                validation_split=0.25,
                )

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:', val_acc[-1])

#1. 그림그려
#2. 튜닝 0.95이상
from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()