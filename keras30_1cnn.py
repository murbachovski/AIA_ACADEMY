from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(7,(2,2),input_shape=(8,8,1)))    # ==> (None, 7, 7, 7)      // # 7 = 7,7,7로 변함 // (2,2)자르는 크기 // input_shape = 이미지의 형태  , , 1 = 흑백  , , 3 = 칼라  
                                # (batch_size, rows, clolumns, channels)
model.add(Conv2D(filters=4, kernel_size=(3,3),activation='relu'))        # 출력 (N, 5, 5, 4)
model.add(Conv2D(10, (2,2)))                                             # 출력 (N, 4, 4, 10)
model.add(Conv2D(5,(2,2)))                                           
model.add(Conv2D(7,(3,3)))                                                   
model.add(Flatten())                        
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()

# 필터 크기(2x2) x 입력 채널(rgb) x 출력 채널 + (bias)=
# 2 x 2 x 1 x 7 + 7 = 35
# 3 x 3 x 7 x 4 + 4 = 256
# 2 x 2 x 4 x 10 + 10 = 170
# 2 x 2 x 10 x 5 + 5 = 205
# 3 x 3 x 5 x 7 + 7 = 322
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 5, 5, 4)           256       
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 10)          170       
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 5)           205       
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 1, 1, 7)           322
# _________________________________________________________________
# flatten (Flatten)            (None, 7)                 0
# _________________________________________________________________
# dense (Dense)                (None, 32)                256
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                330
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 33
# =================================================================
