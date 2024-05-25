import tensorflow as tf
import numpy as np
from tensorflow import keras

# x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
#
# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
# model.fit(x, y, epochs=500)
# print(model.predict([4.0]))
# print(model.predict([5.0]))
# print(model.predict([10.0]))

xs = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0])
ys = np.array([-5.0, -0.0, 5.0, 10.0, 15.0, 20.0])

model = tf.keras.sequential([keras.layers.dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
print(model.predict([100.0]))
print(model.predict([1000.0]))
