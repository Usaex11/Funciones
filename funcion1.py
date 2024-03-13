import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

#funcion:
#f = 3 * np.sin(np.pi * x)
def f(x):
    return 3 * np.sin(np.pi * x)



model = Sequential()
model.add(Dense(200, activation='tanh',  input_shape=(1,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['loss'])
model.summary()

x_train = np.random.uniform(-10, 10, size=(100, 1))
y_train = f(x_train)
x_test = np.linspace(-10, 10, 100).reshape(-1, 1)
y_test = f(x_test)


model.compile(loss='mean_squared_error',optimizer='Adam')
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=10000,
                    verbose=1,
                    validation_data=(x_test, y_test))

a = model.predict(x_test)


plt.plot(history.history["loss"])
plt.figure(figsize=(8, 6))
plt.plot(x_test, f(x_test), label='Función Objetivo', color='blue')
plt.plot(x_test, a, label='Predicciones', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicción de la Red Neuronal')
plt.legend()
plt.grid(True)
plt.show()