import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd


# =============================================== Loading Data ===============================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(np.unique(X_train))

X_train, X_valid, y_train, y_valid = train-test_split(X_train, y_train,
                                                       test_size=0.2, random_state=0)


# =============================================== Modeling =================================================
model = keras.Sequential([
    layers.InputLayer(input_shape(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')),
    layers.MaxPool2D(),

    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])