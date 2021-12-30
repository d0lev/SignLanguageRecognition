import tensorflow as tf
from tensorflow.keras import *
from keras.layers import *
import matplotlib.pyplot as plt
from keras.optimizers import *


class NeuralNetwork:

    def __init__(self, data):
        self.data = data
        self.nn = Sequential()
        self.history = None

    def model(self):
        self.nn.add(Dense(512, input_dim = self.data.features, activation = 'relu'))
        self.nn.add(Dense(512, activation = 'relu'))
        self.nn.add(Dropout(0.1))
        self.nn.add(Dense(self.data.classes, activation = 'softmax'))
        self.nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.history = self.nn.fit(self.data.x_train, self.data.y_train, epochs = 30, batch_size = 50, validation_split = 0.2)
        print(self.nn.evaluate(self.data.x_test, self.data.y_test, verbose = 0))

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(['train set', 'validation set'], loc = 'upper left')
        plt.show()

    def plot_loss_function(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['train set', 'validation set'], loc = 'upper left')
        plt.show()
