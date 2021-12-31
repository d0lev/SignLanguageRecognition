import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , Dropout


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
        self.history = self.cnn.fit(self.x_train, self.data.y_train_cat, epochs = 30, batch_size = 50, validation_data = (self.x_test, self.data.y_test_cat))

        self.prediction = np.argmax(self.cnn.predict(self.x_test), axis=-1)
            # self.cnn.predict_classes(self.x_test)

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

    def plot_confusion_matrix(self):
        self.cm = confusion_matrix(self.data.y_test, self.prediction)
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.set(font_scale=1.6)
        sns.heatmap(self.cm, annot=True, linewidths=.5, ax=ax)
        plt.show()

    def plot_fractional_missclassifications(self):
        incorr_fraction = 1 - np.diag(self.cm) / np.sum(self.cm, axis=1)
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.bar(np.arange(24), incorr_fraction)
        plt.xlabel('True Label')
        plt.ylabel('Fraction of incorrect predictions')
        plt.xticks(np.arange(25), self.data.labels)
        plt.show()