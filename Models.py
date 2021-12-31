import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from NeuralNetwork import *
from ConvolutionalNeuralNetwork import *
tf.compat.v1.disable_eager_execution()


class Data:

    # 27455 training images
    # 7172 test images
    # 25 classes
    # 784 features
    # image represent by (28 * 28) matrix

    def __init__(self):
        self.grayscale = 255
        self.labels = ['A', 'B', 'C', 'D', 'E',
                       'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O',
                       'P', 'Q', 'R', 'S', 'T',
                       'U', 'V', 'W', 'X', 'Y']

        self.classes = len(self.labels)

        self.train_set = np.array(pd.read_csv("sign_mnist_train.csv"))
        self.x_train = (self.train_set[:, 1:] / self.grayscale)
        self.y_train = self.train_set[:, 0]
        self.y_train = keras.utils.to_categorical(self.y_train, self.classes)

        self.test_set = np.array(pd.read_csv("sign_mnist_test.csv"))
        self.x_test = (self.test_set[:, 1:] / self.grayscale)
        self.y_test = self.test_set[:, 0]
        self.y_test = keras.utils.to_categorical(self.y_test, self.classes)

        self.features = (self.train_set.shape[1] - 1)
        print(self.features)

    def plotting_image(self):
        i = random.randint(1, 27455)
        plt.imshow(self.train_set[i, 1:].reshape((28, 28)))
        label_index = self.train_set[i][0]
        plt.title(f"{self.labels[label_index]}")
        plt.show()


dataset = Data()
cnn = ConvolutionalNeuralNetwork(dataset)
cnn.model()