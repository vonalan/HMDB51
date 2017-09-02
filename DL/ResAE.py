# -*- coding: utf-8 -*-

# requirements: 
#   Python 3
#   Numpy 
#   Tensorflow


import numpy as np 
import tensorflow as tf 


class AutoEncoder(object): 
    '''
    An implementation of autoencoder with arbitrary number of hidden layers. 
    '''
    def __init__(self, sizes, activation=None):
        '''
        sizes = [n_in, n_hidden_1, n_hidden_2, ..., n_hidden_m]
        '''
        self.sizes = sizes
        self.learning_rate = 0.1

        self.weights = self._initialize_weights_()
        self.biases = self._initialize_biases_() 

        self.X = tf.placeholder("float", [None, sizes[0]])
        self.encoder = self._encoder_()
        self.decoder = self._decoder_()
        self.Y = self.decoder
        # self.T = tf.placeholder("float", [None, sizes[0]])
        
        self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.X, self.Y), 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.cost)

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

    def transform(self, X):
        valid = self.sess.run(self.encoder, feed_dict={self.X : X})
        return valid

    def saveModel(self, round=None):
        import os
        if not os.path.exists('../model/'):
            os.mkdir('../model/')
        saver = tf.train.Saver()
        saver.save(self.sess, "../model/ae_r%d.ckpt"%round)

    def loadModel(self, round=None):
        saver = tf.train.Saver()
        saver.restore(self.sess, "../model/ae_r%d.ckpt"%round)


class FusedResidualAutoencoder(object):
    '''
    Fused residual autoencoder, inspired by ResNet and GBDT.
    ResNet: residual convolutional neural network,
    GBDT: gradient boosting decision tree
    '''

    def __init__(self, sizes, epoches=20, threshold=0.25, max_width=2):
        self.AEs = list()
        self.sizes = sizes
        self.epoches = epoches
        self.threshold = threshold
        self.max_depth = max_width

    def partial_fit(self, X):
        cost = 0
        depth = 0

        R = X  # residual
        # while(cost > self.threshold and depth < self.max_depth):
        while (depth < self.max_depth):
            ae = AutoEncoder(self.sizes)

            for epoch in range(self.epoches):
                cost = ae.partial_fit(R)  # bug bug bug
                print("depth: %d, epoch: %d, cost: %f" % (depth + 1, epoch + 1, cost))

            _, S = ae.predict(R)
            R = R - S
            # cost = (np.power(R, 2)).mean()
            self.AEs.append(ae)
            depth += 1

        Y = 0
        R = X  # residual
        for ae in self.AEs:
            _, S = ae.predict(R)
            Y += S
            R = R - S
            # cost = (np.power(R, 2)).mean()

        cost = (np.power(X - Y, 2)).mean()

        return cost

    def predict(self, X):
        Y = 0
        residual = X
        for ae in self.AEs:
            Y = ae.predict(residual)
            residual = X - Y
            cost = (np.power(residual, 2)).mean()

        return Y

    def transform(self, X):
        M = 0
        residual = X
        for ae in self.AEs:
            M += ae.transform(residual)
            Y = ae.predict(residual)
            residual = X - Y
            # cost = (np.power(residual, 2)).mean()

        return M

    def saveModel(self, round=None):
        pass

    def loadModel(self, round=None):
        pass


class StackedResidualAutoencoder(object):
    '''
    Stacked residual autoencoder, inspired by ResNet and GBDT. 
    ResNet: residual convolutional neural network, 
    GBDT: gradient boosting decision tree
    '''

    def __init__(self, sizes, epoches=20, threshold=0.25, max_depth=2):
        self.AEs = list()
        self.sizes = sizes 
        self.epoches = epoches
        self.threshold = threshold
        self.max_depth = max_depth

    def partial_fit(self, X):
        cost = 0
        depth = 0

        R = X # residual
        # while(cost > self.threshold and depth < self.max_depth):
        while(depth < self.max_depth): 
            ae = AutoEncoder(self.sizes)

            for epoch in range(self.epoches): 
                cost = ae.partial_fit(R) # bug bug bug
                print("depth: %d, epoch: %d, cost: %f"%(depth+1, epoch+1, cost))

            _, S = ae.predict(R)
            R = R - S
            # cost = (np.power(R, 2)).mean()
            self.AEs.append(ae)
            depth += 1

        Y = 0
        R = X # residual
        for ae in self.AEs:
            _, S = ae.predict(R)
            Y += S
            R = R - S
            # cost = (np.power(R, 2)).mean()

        cost = (np.power(X-Y,2)).mean()

        return cost

    def predict(self, X):
        Y = 0
        residual = X 
        for ae in self.AEs:
            Y = ae.predict(residual)
            residual = X - Y
            cost = (np.power(residual, 2)).mean()

        return Y

    def transform(self, X):
        M = 0 
        residual = X 
        for ae in self.AEs:
            M += ae.transform(residual)
            Y = ae.predict(residual)
            residual = X - Y
            # cost = (np.power(residual, 2)).mean()
        
        return M 
    
    def saveModel(self, round=None): 
        pass 
    
    def loadModel(self, round=None): 
        pass 

if __name__ == '__main__': 
    import utils
    img = utils.read_data_from_image('imgtest.png')
    print(img.shape)
    img = img.reshape((-1,img.shape[0]*img.shape[1]))
    print(img.shape)
    img = img/255.0

    # ae = AutoEncoder([img.shape[1], 1024])
    # for epoch in range(20000): 
    #     cost = ae.partial_fit(img)
    #     print("epoch: %d, cost: %f"%(epoch+1, cost))

    srae = StackedResidualAutoencoder([img.shape[1], 1024], epoches=20000, threshold=0.25, max_depth=2)
    cost = srae.partial_fit(img)
    print(cost)