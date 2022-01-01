import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , Dropout
from keras.models import Model


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

    def plot_filters(self):
        filters, biases = self.cnn.layers[6].get_weights()
        fig1 = plt.figure(figsize=(8, 12))
        n_filters = 128
        for i in range(1, n_filters + 1):
            f = filters[:, :, :, i - 1]
            fig1 = plt.subplot(8, 16, i)
            fig1.set_xticks([])  # Turn off axis
            fig1.set_yticks([])
            plt.imshow(f[:, :, 0], cmap='gray')  # Show only the filters from 0th channel (R)
            # ix += 1
        plt.show()

    def plot_features_maps(self):
        layer = self.cnn.layers
        img = self.x_test[5].reshape(1, 28, 28, 1)
        layer_outputs = [layer.output for layer in self.cnn.layers]
        activation_model = Model(inputs=self.cnn.input, outputs=layer_outputs)
        activations = activation_model.predict(img)
        layer_names = []
        for layer in self.cnn.layers[0:6]:
            layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

        images_per_row = 16
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                    :, :,
                                    col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,  # Displays the grid
                    row * size: (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()