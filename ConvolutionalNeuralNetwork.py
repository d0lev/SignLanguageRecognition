import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import *


class ConvolutionalNeuralNetwork:

    def __init__(self, dataset):
        self.history = None
        self.data = dataset
        self.train_examples = dataset.x_train.shape[0]
        self.test_examples = dataset.x_test.shape[0]
        self.x_train = dataset.x_train.reshape(self.train_examples, *(28, 28, 1))
        self.x_test = dataset.x_test.reshape(self.test_examples, *(28, 28, 1))
        self.cnn = Sequential()

    def model(self):
        self.cnn.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
        self.cnn.add(MaxPooling2D(pool_size = (2, 2)))
        self.cnn.add(Dropout(0.2))

        self.cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
        self.cnn.add(MaxPooling2D(pool_size = (2, 2)))
        self.cnn.add(Dropout(0.2))

        self.cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(0.2))

        self.cnn.add(Flatten())
        self.cnn.add(Dense(128, activation = 'relu'))
        self.cnn.add(Dense(self.data.classes, activation= 'softmax'))

        self.cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.history = self.cnn.fit(self.x_train, self.data.y_train, epochs = 30, batch_size = 50, validation_data = (self.x_test, self.data.y_test))

        prediction = self.cnn.predict(self.x_test)

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

    def plot_loss_function(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['train set', 'validation set'], loc = 'upper left')
        plt.show()
