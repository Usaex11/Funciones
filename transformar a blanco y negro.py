import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math


imagen_camino = "/Users/esasm/Documents/Redes Neuronales/Blanco y negro/motoren-test-2.jpg"
imagen = tf.keras.preprocessing.image.load_img(imagen_camino, target_size=(150, 150))
imagen_array = tf.keras.preprocessing.image.img_to_array(imagen)
imagen_tensor = tf.expand_dims(imagen_array, 0)


class BlancoYNegro(tf.keras.layers.Layer):
    def __init__(self):
            super(BlancoYNegro, self).__init__()

    def call(self, inputs):
        blanynegro=tf.image.rgb_to_grayscale(inputs)
        return blanynegro

model_F = tf.keras.Sequential()
model_F.add(BlancoYNegro())
model_F.summary()

#x=model_F.predict(image_tensor)
#imagen_bn_np = x.numpy().squeeze() 
imagen_bn = model_F.predict(imagen_tensor).squeeze()
plt.imshow(imagen_bn, cmap='gray')
plt.title("Imagen en blanco y negro")
plt.show()



