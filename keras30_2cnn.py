from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()                   # (N, 3)
model.add(Dense(10, input_shape=(3,))) # (batch_szie, input_dim)
model.add(Dense(units=15))             # 출력 (batch_size, units)
model.summary()

#batch_size, input_dim
#batch_size, units

# tf.keras.layers.Dense(
#     units, output, filters
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# units: Positive integer, dimensionality of the output space.
# activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
# use_bias: Boolean, whether the layer uses a bias vector.
# kernel_initializer: Initializer for the kernel weights matrix.
# bias_initializer: Initializer for the bias vector.
# kernel_regularizer: Regularizer function applied to the kernel weights matrix.
# bias_regularizer: Regularizer function applied to the bias vector.
# activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
# kernel_constraint: Constraint function applied to the kernel weights matrix.
# bias_constraint: Constraint function applied to the bias vector.