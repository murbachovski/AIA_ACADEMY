from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
from sklearn.metrics import accuracy_score 

(x_train, y_train), (x_test, y_test) = mnist.load_data() # python 기초 문법
# y는 뽑지 않고 x만 뽑아서 사용하겠다.

# x = np.concatenate((x_train, x_test), axis=1)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
temp = ['154', '331', '486', '713']
# print(x.shape)
for i in temp:
    # perform PCA
    pca = PCA(n_components=i)
    x = pca.fit_transform(x)

    # build and train model
    model = Sequential([
        Dense(64, input_shape=i,),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=5)

    # evaluate model on test data
    y_pred = np.argmax(model.predict(x), axis=-1)
    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

#############################################
#############################################

# ACC :
# 나의 최고의 CNN : acc: 0.7226999998092651
# 나의 최고의 DNN : 
# PCA 0.95  :
# PCA 0.99  :
# PCA 0.999 :
# PCA 1.0   :





# pca_EVR = pca.explained_variance_ratio_
# # print(pca_EVR)

# cumsum = np.cumsum(pca_EVR)
# # print(np.cumsum(pca_EVR))

# print('1.0', np.argmax(cumsum >= 1.0) + 1)
# print('0.95', np.argmax(cumsum >= 0.999) + 1)
# print('0.9', np.argmax(cumsum >= 0.99) + 1)
# print('0.9', np.argmax(cumsum >= 0.95) + 1)
# # 1.0 713
# # 0.95 486
# # 0.9 331
# # 0.9 154

