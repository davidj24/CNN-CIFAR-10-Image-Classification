import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd


# =============================================== Loading Data ===============================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                       test_size=0.2, random_state=0)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_image(dataset, labels, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(dataset[index])
    plt.xlabel(classes[labels[index][0]])


# =============================================== Modeling =================================================
model = keras.Sequential([
    layers.InputLayer(input_shape=(32, 32, 3)),

    layers.RandomRotation(factor=.1),
    layers.RandomFlip(mode='horizontal'),


    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same'),
    # layers.MaxPool2D(),
    # layers.Dropout(.3),

    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(.3),
    layers.Dense(10, activation='softmax')
    ])
    

early_stopping = EarlyStopping(
    patience=4,
    min_delta=.001,
    restore_best_weights=True
    )

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
    )


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping],
    epochs=100
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()

# model.save('CIFAR-10-MODEL.keras')

# model = keras.model.load_model('CIFAR-10-MODEL.keras')



loss, accuracy = model.evaluate(X_test, y_test)
print(f'loss: {loss}')
print(f'accuracy: {accuracy}')

# Current scores:
# loss: 0.7955142259597778
# accuracy: 0.7235999703407288


#Following with video 3 total layers
# loss: 0.8620476722717285
# accuracy: 0.708899974822998

# With Data Augmentation 3 total layers
# loss: 0.7880503535270691
# accuracy: 0.7373999953269958

# Dropout layers with only 3 total and 64 neurons in the head dense
# loss: 0.7983840107917786
# accuracy: 0.7282000184059143

# Dropout layers with only 3 total and 64 neurons in the head dense
# loss: 0.6357991099357605
# accuracy: 0.786899983882904

# All layers, no maxPooling on last layer, and no dropout at all
# loss: 0.5952411890029907
# accuracy: 0.8084999918937683

# Same as above but 256 neurons in the head