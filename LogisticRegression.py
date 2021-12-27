import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import sys
tf.compat.v1.disable_eager_execution()


class LogisticRegression:
    def __init__(self, dataset):
        self.dataset = dataset

    def model(self):
        batch_size = 270
        features = self.dataset.features
        classes = self.dataset.classes
        # Size of x -> (?, 784)
        x = tf.placeholder(tf.float32, [None, features])
        # Size of y_ -> (?, 25)
        y_ = tf.placeholder(tf.float32, [None, classes])

        W = tf.Variable(tf.zeros([features, classes]))
        b = tf.Variable(tf.zeros([classes]))

        z = tf.matmul(x, W) + b
        # Define activation function - Softmax
        pred = tf.nn.softmax(tf.matmul(x, W) + b)

        # Define the loss function - Cross entropy
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=z))

        # Define the update function - Gradient descent
        update = tf.train.AdamOptimizer(0.001).minimize(loss)

        # Define accuracy
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.loss_trace = []
        self.train_acc = []
        self.test_acc = []

        for epoch in range(15):
            for row in range(100):
                batch_x = self.dataset.x_train[row * batch_size : (row + 1) * (batch_size-1)]
                batch_y = self.dataset.y_train[row * batch_size : (row + 1) * (batch_size-1)]

                _, loss_validation = sess.run([update, loss], feed_dict={x: batch_x, y_: batch_y})
                temp_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y_: batch_y})
            temp_test_acc = sess.run(accuracy, feed_dict={x: self.dataset.x_test, y_: self.dataset.y_test})
            self.loss_trace.append(temp_loss)
            self.train_acc.append(temp_train_acc)
            self.test_acc.append(temp_test_acc)
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch, temp_loss, temp_train_acc, temp_test_acc))


    def plotLoss(self):
        plt.plot(self.loss_trace)
        plt.title('Cross Entropy Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def plotAccuracy(self):
        plt.plot(self.train_acc, 'b-', label='train accuracy')
        plt.plot(self.test_acc, 'k-', label='test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Train and Test Accuracy')
        plt.legend(loc='best')
        plt.show()