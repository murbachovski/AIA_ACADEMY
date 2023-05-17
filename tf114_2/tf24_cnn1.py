from keras.datasets import mnist
import keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import time
tf.random.set_random_seed(337)

# 1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

# 실습 맹그러!
x_train = x_train.reshape(60000, 28, 28, 1)/255.
# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 784) (10000, 10)

# 2. MODEL
x = tf.compat.v1.placeholder('float', [None, 28, 28, 1])
y = tf.compat.v1.placeholder('float', [None, 10])

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1')
b1 = tf.Variable(tf.zeros([32]), name='b1')
# layer1 = tf.compat.v1.matmul(x, w1) + b1 # model.add(Dense()) = same
layer1 = tf.compat.v1.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

w2 = tf.Variable(tf.random_normal([3, 3, 64, 32]), name='w2')
b2 = tf.Variable(tf.zeros([32]), name='b2')
layer2 = tf.nn.relu(tf.compat.v1.matmul(L1_maxpool, w2) + b2, padding= 'VALID')
layer2 += b2
L2_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

w3 = tf.Variable(tf.random_normal([3, 3, 64, 16]), name='w3')
b3 = tf.Variable(tf.zeros([16]), name='b2')
layer3 = tf.nn.relu(tf.compat.v1.matmul(L2_maxpool, w3) + b3, padding= 'VALID')
layer3 += b3
L3_maxpool = tf.nn.max_pool2d(layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

#FLATTEN
L_flat = tf.reshape(layer3, [-1, 6*6*16])

## Dense레이어 연결하고 있어!!

# layer4 DNN
w4 = tf.Variable(tf.random_normal([6*6*16, 100]), name='w4')
b4 = tf.Variable(tf.zeros([100]), name='b4')
layer4 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w4) + b4)
layer4 = tf.nn.dropout(layer4, rate=0.3)

w5 = tf.Variable(tf.random_normal([100, 10]), name='w5')
b5 = tf.Variable(tf.zeros([100]), name='b5')
hypothesis = tf.nn.relu(tf.compat.v1.matmul(layer4, w5) + b5)
hypothesis = tf.nn.softmax(hypothesis)

# 3. COMPILE, FIT
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y), axis=1)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100

epochs = 20
total_batch = int(len(x_train)/batch_size) # 60000/100 = 600
start_time = time.time()

for step in range(epochs):
    avg_cost = 0
    for i in range(int(total_batch)):       # 100개씩 600번 돌려
        start = i * batch_size              # 0, 100, 200 ... 59900
        end = start + batch_size            # 100, 200, 300 ... 60000
        
        # x_train[:100], y_train[:100]

        cost_val, _, w_val, b_val = sess.run([train, loss], feed_dict={x:x_train[start:end], y:y_train[start:end]})
        cost_val += cost_val / total_batch
    print("EPOCHS:", step + 1, 'LOSS: {:.9f}'.format(avg_cost))

    if step % 20 == 0:
        print(step, cost_val)
end_time = time.time()
print("DONE")

# 4. PREDICT
y_predict = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x:x_test})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))

y_test_arg = np.argmax(y_test,1)
 
acc = accuracy_score(y_predict_arg, y_test_arg)
print("accuracy_score:", acc)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=np.float32))

acc = sess.run([hypothesis, predicted, accuracy],
                    feed_dict={x:x_test, y:y_test})

print("ACC: ",  acc)
print("TIME: ", end_time - start_time)

