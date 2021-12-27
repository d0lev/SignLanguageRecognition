import tensorflow as tf
from tensorflow.keras import *
from keras.layers import *
from keras.optimizers import *


class NeuralNetwork:

    def __init__(self, data):
        self.data = data

    def model(self):
        nn = Sequential()
        nn.add(Dense(480, input_dim = self.data.features, activation = 'relu'))
        nn.add(Dropout(0.2))
        nn.add(Dense(320, activation = 'relu'))
        nn.add(Dense(self.data.classes, activation = 'softmax'))
        nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        nn.fit(self.data.x_train, self.data.y_train, epochs = 30, batch_size = 50, validation_split = 0.2)
        print(nn.evaluate(self.data.x_test, self.data.y_test, verbose = 0))

    def plot_accuracy(self):
        pass

    def plot_loss_function(self):
        pass
