import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import warnings


x_train = [1, 2, 3]  # Input data
y_train = [1, 2, 3]  # Output data
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w
loss = tf.reduce_mean(tf.square(hypothesis - y))  # Mean Squared Error (MSE)

lr = 1
gradient = tf.reduce_mean((x * w - y) * x)
descent = w - lr * gradient
update = w.assign(descent)  # w = w - lr * gradient

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x: x_train, y: y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
y_pred = sess.run(hypothesis, feed_dict={x: x_train}) 
r2 = r2_score(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)

print("R2 Score:", r2)
print("MAE:", mae)

sess.close()
