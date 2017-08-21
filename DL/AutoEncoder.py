# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

class AutoEncoder(object): 
    def __init__(self, sizes, activation=None): 
        '''
        sizes = [n_in, n_hidden_1, n_hidden_2, ..., n_hidden_m]
        '''

        self.sizes = sizes 
        self.weights = self._initialize_weights_()
        self.biases = self._initialize_biases_() 

        self.X = tf.placeholder("float", [None, sizes[0]])
        self.encoder = self._encoder_()
        self.decoder = self._decoder_()
        self.Y = self.decoder
        # self.T = tf.placeholder("float", [None, sizes[0]])
        
        self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.X, self.Y), 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(self.cost)

        # only works in namespace and scope of AutoEncoder
        self.init = tf.global_variables_initializer() # may be a bug
        self.sess = tf.Session()
        self.sess.run(self.init)
        print("Initialization finished! ")
    
    def _initialize_weights_(self): 
        weights = dict() 

        # encoder 
        for i in range(len(self.sizes)-1): 
            key = 'encoder_w%d'%(i+1)
            val = tf.Variable(tf.random_normal([self.sizes[i], self.sizes[i+1]]))
            weights[key] = val
        
        # decoder 
        for i in range(1, len(self.sizes)): 
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
            biases[key] = val
        
        # decoder 
        for i in range(1, len(self.sizes)): 
            key = 'deconder_b%i'%i 
            val = tf.Variable(tf.random_normal([self.sizes[-(i+1)]]))
            biases[key] = val

        return biases 

    # encoder
    def _encoder_(self): 
        layer = self.X 
        for i in range(len(self.sizes)-1): 
            key_ws = 'encoder_w%d'%(i+1)
            key_bs = 'encoder_b%d'%(i+1)
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights[key_ws]),
                                   self.biases[key_bs]))
        return layer
    
    # decoder
    def _decoder_(self): 
        layer = self.encoder  

        for i in range(1, len(self.sizes)): 
            key_ws = 'deconder_w%i'%i
            key_bs = 'deconder_b%i'%i 
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights[key_ws]),
                                   self.biases[key_bs]))
    
        return layer 
    
    # public interface
    def partial_fit(self, X):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict={self.X : X})
        return cost
    
    def predict(self, X):
        cost, valid = self.sess.run([self.cost, self.Y], feed_dict={self.X : X})
        return cost, valid

if __name__ == '__main__': 
    ae = AutoEncoder([64,32,16])
