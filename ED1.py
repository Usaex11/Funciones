import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

batch_size=450

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)])

optimizer = tf.keras.optimizers.Adam()

def train_step(model, x):
    with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            y_pred = model(x)
        dy_dx = tape2.gradient(y_pred, x)
        eq = (dy_dx * x) + y_pred + (x*x*tf.cos(x))
        x_o = tf.zeros((batch_size, 1))
        y_o = model(x_o)
        ic = 0.
        loss = loss_function(ic, y_o) + loss_function(0.0, eq)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_step2(model, x):
    with tf.GradientTape() as tape:
        y_pred = model(x)
    dy_dx = tape.gradient(y_pred, x)
    eq = (dy_dx * x) + y_pred + (x*x*tf.cos(x))
    x_o = tf.zeros((batch_size, 1))
    y_o = model(x_o)
    ic = 0.
    loss = loss_function(ic, y_o) + loss_function(0.0, eq)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

losses = []
model.compile(optimizer=optimizer, loss=loss_function)
epochs = 2500

for epoch in range(epochs):
    x_train = x = tf.random.uniform(shape=(batch_size, 1), minval=-5.0, maxval=5.0, dtype=tf.float32)
    loss = train_step(model, x)
    losses.append(loss)

x_v = np.linspace(-5, 5, 100)
y_pred_values = model.predict(x_v.reshape(-1, 1))
solucion = ((((x_v*x_v)+2)*np.sin(x_v)+1)/x_v) + 2*np.cos(x_v)

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