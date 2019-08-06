from __future__ import absolute_import, division, print_function, \
    unicode_literals
from tensorflow.python.keras import layers, models, datasets, utils

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(
    y_train, 10), utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
