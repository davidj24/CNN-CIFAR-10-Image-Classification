import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from tensorflow.keras.datasets import cifar10

# =============================================== Loading Data ===============================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(np.unique(y_train))