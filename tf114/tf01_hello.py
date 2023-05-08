import tensorflow as tf
import warnings
print(tf.__version__)


print('Hello world')

aaa = tf.constant('hello world')
print(aaa)  # Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session() 동일하지만 버전 차이가 있다.
sess = tf.compat.v1.Session()
print(sess.run(aaa))
