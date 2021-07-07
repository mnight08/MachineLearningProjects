from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import pandas_profiling
from tensorflow.keras.utils import plot_model


mnist=tf.keras.datasets.mnist
(x_tr, y_tr), (x_t, y_t) = mnist.load_data()

x_tr, x_t = x_tr / 255.0, x_t / 255.0

model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(10, activation='softmax')
        ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_tr,y_tr, epochs=20)
plot_model(model, show_shapes = True)
model.evaluate(x_t,y_t)

