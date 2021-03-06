# -*- coding: utf-8 -*-

class AutoEncoder(object): 
    def __init__(self, sizes, activation=tf.nn.sigmoid): 
        '''
        sizes = [n_in, n_hidden_1, n_hidden_2, ..., n_hidden_m]
        '''

        self.sizes = sizes 
        self.weights = self._initialize_weights_()
        self.bias = self._initialize_biases_() 

        self.X = tf.placeholder("float", [None, sizes[0]])
        # self.T = tf.placeholder("float", [None, sizes[0]])
        self.Y = self._nn_out_()

        self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.X, self.Y), 2)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        self.init = tf.global_variables_initializer() # may be a bug
        self.sess = tf.Session()
        self.sess.run(self.init)
    
    def _initialize_weights_(self): 
        weights = dict() 

        # encoder 
        for i in range(len(self.sizes)-1): 
            key = 'encoder_w%d'%(i+1)
            val = tf.Variable(tf.random_normal([self.sizes[i], self.sizes[i+1]]))
            weights[key] = val
        
        # decoder 
        for i in range(1, len(sizes)+1): 
            key = 'deconder_w%i'%i 
            val = tf.Variable(tf.random_normal([self.sizes[-i], self.sizes[-(i+1)]]))
            weights[key] = val

        return weights 

    def _initialize_biases_(self): 
        biases = dict() 

        # encoder 
        for i in range(len(self.sizes)-1): 
            key = 'encoder_b%d'%(i+1)
            val = tf.Variable(tf.random_normal([self.sizes[i+1]]))
            weights[key] = val
        
        # decoder 
        for i in range(1, len(sizes)+1): 
            key = 'deconder_b%i'%i 
            val = tf.Variable(tf.random_normal([self.sizes[-(i+1)]]))
            weights[key] = val

        return bias 

    
    
    # encoder
    def _encoder_(self, x): 
        layer = None 
        for i in range(len(self.sizes)-1): 
            key_ws = 'encoder_w%d'%(i+1)
            key_bs = 'encoder_b%d'%(i+1)
            layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights[key_ws]),
                                   self.biases[key_bs]))
        return layer

    
    # decoder
    def _decoder_(self, x): 
        layer = None  

        for i in range(1, len(self.sizes)+1): 
            key_ws = 'deconder_w%i'%i
            key_bs = 'deconder_b%i'%i 
            layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights[key_ws]),
                                   self.biases[key_bs]))
    
        return layer 
    
    # output
    def __nn__out__(self, x):
        layer = x 
        layer = self._encoder_(layer)
        layer = self._decoder_(layer)

        return layer

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 2000
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 4096 # 1st layer num features
n_hidden_2 = 1024 # 2nd layer num features
n_input = 60 * 80 * 10 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Initialization Finished!")

    import DataLoader
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        total_batch = 1 
        for i in range(total_batch):
            batch_xs = DataLoader.load()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    # encode_decode = sess.run(
    #     y_pred, feed_dict={X:batch_xs})
    
    # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()
