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
x_train = x_train.reshape(60000, 28*28)/255.
# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 784) (10000, 10)

# 2. MODEL
x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', [None, 10])

w1 = tf.Variable(tf.random_normal([784, 128]), name='w1')
b1 = tf.Variable(tf.zeros([128]), name='b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
dropout1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.Variable(tf.random_normal([128, 64]), name='w2')
b2 = tf.Variable(tf.zeros([64]), name='b2')
layer2 = tf.nn.relu(tf.compat.v1.matmul(dropout1, w2) + b2)

w3 = tf.Variable(tf.random_normal([64, 32]), name='w3')
b3 = tf.Variable(tf.zeros([32]), name='b3')
layer3 = tf.nn.selu(tf.compat.v1.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([32, 10]), name='w4')
b4 = tf.Variable(tf.zeros([10]), name='b4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

# 3. COMPILE, FIT
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y), axis=1)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100

epochs = 100
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

