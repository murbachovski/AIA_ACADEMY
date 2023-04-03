from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000
np.random.seed(0)
randidx = np.random.randint(60000, size=40000)
randidx = np.random.randint(x_train.shape[0], size=augment_size) 

print(randidx)
print(randidx.shape) # (40000,)
print(np.min(randidx), np.max(randidx)) # 5 59997

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy() # copy 안 넣으면 과적합(중복)
print(x_augmented)
# print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)
print(y_augmented)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2], 1)
x_augmented = x_augmented.reshape(
                        x_augmented.shape[0],
                        x_augmented.shape[1],
                        x_augmented.shape[2], 1  
)
# x_augmented = train_datagen.flow(
#     x_augmented,
#     y_augmented,
#     batch_size=augment_size,
#     shuffle=False
# )

x_augmented = train_datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

print(x_augmented)
print(x_augmented.shape) # (40000, 28, 28, 1)

# x_train = x_train + x_augmented
# print(x_train)

x_train = np.concatenate((x_train/255., x_augmented))
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255. # scaler 맞춰줍니다.

print(x_train.shape, y_train.shape)
print(np.max(x_train), np.min(x_train)) # 255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0
print(y_train.shape)

# 모델 만들어봐

#2. MODEL
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,))

#3. COMPILE
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train)

#4. PREDICT
results = model.evaluate(x_test, y_test)
print(results)