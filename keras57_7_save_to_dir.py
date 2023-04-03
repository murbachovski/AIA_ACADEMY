from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
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

augment_size = 1000
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

start_time = time.time()
print("START")
x_augmented = train_datagen.flow( #증폭
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir='d:/temp/'
).next()[0]
end_time = time.time() - start_time

print(augment_size, '개 증폭에 걸린시간 : ', round(end_time,2),  '초')









# print(x_augmented)
# print(x_augmented.shape) # (40000, 28, 28, 1)

# # x_train = x_train + x_augmented
# # print(x_train)

# x_train = np.concatenate((x_train/255., x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
# x_test = x_test/255. # scaler 맞춰줍니다.

# print(x_train.shape, y_train.shape)
# print(np.max(x_train), np.min(x_train)) # 255.0 0.0
# print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0
