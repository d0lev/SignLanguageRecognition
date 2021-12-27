import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

tf.compat.v1.disable_eager_execution()


class Data:

    # 27455 training images
    # 7172 test images
    # 25 classes
    # 784 features
    # image represent by (28 * 28)

    def __init__(self):
        self.train_set = np.array(pd.read_csv("sign_mnist_train.csv"))
        self.test_set = np.array(pd.read_csv("sign_mnist_test.csv"))
        self.classes = ['A', 'B', 'C', 'D', 'E',
                        'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O',
                        'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y']

        self.features = (self.train_set.shape[1] - 1)

    def plotting_image(self):
        i = random.randint(1, 27455)
        plt.imshow(self.train_set[i, 1:].reshape((28, 28)))
        label_index = self.train_set[i][0]
        plt.title(f"{self.classes[label_index]}")
        plt.show()


dataset = Data()
dataset.plotting_image()
