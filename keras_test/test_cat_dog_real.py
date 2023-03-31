from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import time
from keras.callbacks import EarlyStopping
path = 'd:/study_data/_data/cat_dog/PetImages/'
save_path = 'd:/study_data/_save/cat_dog/'
# 경로 변수 저장해줘
datagen = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

cat_dog = datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=500,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)
# 데이터 가져와
cat_dog_x = cat_dog[0][0]
cat_dog_y = cat_dog[0][1]

end = time.time()
print(end - start)

cat_dog_x_train, cat_dog_x_test, cat_dog_y_train, cat_dog_y_test = train_test_split(
    cat_dog_x,
    cat_dog_y,
    test_size=0.025,
    shuffle=True,
    random_state=2222
)

# print(cat_dog_x_train.shape)
# print(cat_dog_x_test.shape)
# print(cat_dog_y_train.shape)
# print(cat_dog_y_test.shape)

np.save(save_path + 'keras56_cat_dog_x_train.npy', arr=cat_dog_x_train)
np.save(save_path + 'keras56_cat_dog_x_test.npy', arr=cat_dog_x_test)
np.save(save_path + 'keras56_cat_dog_y_train.npy', arr=cat_dog_y_train)
np.save(save_path + 'keras56_cat_dog_y_test.npy', arr=cat_dog_y_test)

#2. MODEL
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(32, (2, 2),activation='relu'))
model.add(Conv2D(16, (2, 2),activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=800,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(cat_dog_x_train, cat_dog_y_train, epochs=1000, validation_data=(cat_dog_x_test, cat_dog_y_test), callbacks=[Es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()