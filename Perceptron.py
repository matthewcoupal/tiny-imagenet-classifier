from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

# Imports
#import os
import tensorflow as tf
#import pandas as pd
import numpy as np
from DataSet import DataSet


class Perceptron(object):
    """A simple perceptron.

    Input Nodes -Weights-> Output Nodes
    Should not have any hidden layers

    Attributes:
        inputs: n-tensor representing n inputs
        outputs: m-tensor representing m outputs

    """
    seed = 128
    random_number_generator = np.random.RandomState(seed)

    def __init__(self, features, labels, label_set, mode="train"):
        """

        :param features: Numpy array of images 64x64x3
        :param labels: Numpy array of corresponding labels
        :param label_set: Set of label names
        :param mode: train or validate
        """



        #data_tf = tf.convert_to_tensor(features)
        assert features.shape[0] == labels.shape[0]

        ds = DataSet(features, labels)

        #tf_features = tf.convert_to_tensor(features, name="Features")
        #tf_labels = tf.convert_to_tensor(labels, name="Labels")

        #tf_data_set = [tf_features, tf_labels]

        x = tf.placeholder(tf.float32, [None, 4096])
        W = tf.Variable(tf.zeros([4096,len(label_set)]))
        b = tf.Variable(tf.zeros([len(label_set)]))
        y = tf.matmul(x, W) + b

        y_ = tf.placeholder(tf.string, [None, len(label_set)])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        #for i in range(int(len(labels)/100)):
        #  xs  = ds.features[i*100:i*100 + 100]
        #   ys = ds.labels[i*100:100*i+100]

           #assert all(x.shape == (64, 64) for x in xs)
        #   session.run(train_step, feed_dict={x: xs, y_: ys})







        # Create the input layer
        # -1, number of pictures set dynamically based on the size of features
        # 64x64 image
        # 3 for RGB
        # self.input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])


    def train(self):
        pass

    def test(self):
        pass

    def next_batch(self):
        pass
