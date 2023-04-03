import  numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

#1. DATA
path = 'd:/study_data/_data/dogs_breed/'
save_path = 'd:/study_data/_save/dogs_breed/'
np.random.seed(1111)
all_data = ImageDataGenerator(
    rescale=1./255
)

all_data2 = ImageDataGenerator(
    rescale=1./1
)

start = time.time()

dogs_breed_data = all_data.flow_from_directory(
    path,
    target_size=(50, 50),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

x_train = dogs_breed_data[0][0]
y_train = dogs_breed_data[0][1]

end = time.time()

augment_size = 425
batch_size=64
randidx=np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()      
y_augmented = y_train[randidx].copy() 

x_train = x_train.reshape(x_train.shape[0], 
                          x_train.shape[1], 
                          x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2],
                        1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)

x_augmented = all_data.flow(
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False,
).next()[0] 

x_train = np.concatenate((x_train/255., x_augmented))   
y_train = np.concatenate((y_train, y_augmented))   
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_test = x_test/255.

xy_train = all_data2.flow(x_train, y_train,
                               batch_size = batch_size, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(
    x_dogs_breed,
    y_dogs_breed,
    test_size=0.25,
    random_state=1111
)

print(x_train.shape, x_test.shape) # (75, 50, 50, 3) (25, 50, 50, 3)
print(y_train.shape, y_test.shape) # (75, 5) (25, 5)

np.save(save_path + 'keras58_dogs_breed_x_train.npy', arr=x_train)
np.save(save_path + 'keras58_dogs_breed_x_test.npy', arr=x_test)
np.save(save_path + 'keras58_dogs_breed_y_train.npy', arr=y_train)
np.save(save_path + 'keras58_dogs_breed_y_test.npy', arr=y_test)