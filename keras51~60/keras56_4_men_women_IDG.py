import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
path = 'd:/study_data/_data/men_women/'
save_path = 'd:/study_data/_save/men_women/'

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

people_data = all_data.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_data = people_data[0][0]
y_data = people_data[0][1]

end = time.time()


print(end - start)

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.025,
    random_state=2222,
    shuffle=True
)

np.save(save_path + 'keras56_mw_x_train.npy', arr=x_train)
np.save(save_path + 'keras56_mw_x_test.npy', arr=x_test)
np.save(save_path + 'keras56_mw_y_train.npy', arr=y_train)
np.save(save_path + 'keras56_mw_y_test.npy', arr=y_test)

print(x_train.shape, x_test.shape) # (97, 100, 100, 3) (3, 100, 100, 3)
print(y_train.shape, y_test.shape) # (97,) (3,)