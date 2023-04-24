from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data() # python 기초 문법
# y는 뽑지 않고 x만 뽑아서 사용하겠다.

# x1 = x_train, x_test
# print(x1)

# x = np.concatenate((x_train, x_test), axis=1)
x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

# print(x.shape)
pca = PCA(n_components=784) 
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(np.cumsum(pca_EVR))

print('1.0', np.argmax(cumsum >= 1.0) + 1)
print('0.95', np.argmax(cumsum >= 0.999) + 1)
print('0.9', np.argmax(cumsum >= 0.99) + 1)
print('0.9', np.argmax(cumsum >= 0.95) + 1)


