import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np


def loss_function(y_real, y_pred):
    return tf.reduce_mean(tf.square(y_real-y_pred))

batch_size=400

model = tf.keras.Sequential()
model.add(Input(shape=(1,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))


optimizer = tf.keras.optimizers.Adam()

def train_step(model, x):
    with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape3:
                tape3.watch(x)
                y_pred = model(x)
                dy_dx = tape2.gradient(y_pred, x)
                d2y_dx2 = tape3.gradient(dy_dx, x)
                eq = d2y_dx2 + y_pred
                x_o = tf.zeros((batch_size, 1))
                y_o = model(x_o)
                ic = 0.
                x_o2 = tf.zeros((batch_size, 1))
                y_o2 = model(x_o2)
                ic2  = 0.5
                loss = loss_function(ic, y_o) + loss_function(0.0, eq) + loss_function(ic2, dy_dx)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

losses = []
model.compile(optimizer=optimizer, loss=loss_function)
epochs = 1500

for epoch in range(epochs):
    x_train = x = tf.random.uniform(shape=(batch_size, 1), minval=-5.0, maxval=5.0, dtype=tf.float32)
    loss = train_step(model, x)
    losses.append(loss)

x_v = np.linspace(-5, 5, 100)
y_pred_values = model.predict(x_v.reshape(-1, 1))
solucion = np.cos(x_v) - (0.5*np.cos(x_v))

plt.plot(losses)
plt.xlabel('Iteración de entrenamiento')
plt.ylabel('Pérdida')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.show()

plt.plot(x_v, y_pred_values, label='Aproximación')
plt.plot(x_v, solucion, label='Solución Exacta')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproximación y Solución Exacta de la Ecuación Diferencial')
plt.legend()
plt.show()