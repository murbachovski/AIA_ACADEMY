from tensorflow.keras.datasets import fashion_mnist
import random
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)    
print(x_test.shape, y_test.shape)       
  
import matplotlib.pyplot as  plt
# random_x = random.choice(x_train)
# plt.imshow(random_x)
# plt.show()

while x_train.shape[2] > 30:
    random_x = random.choice(x_train)
    plt.imshow(random_x)
    plt.show()
else:
    random_x = random.choice(x_train)
    plt.imshow(random_x)
    plt.show()