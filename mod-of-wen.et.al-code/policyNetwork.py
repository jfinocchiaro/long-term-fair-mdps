#-*-coding:utf-8-*-

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import initializers

import numpy as np


""" Policy network
    Input: [0,1] x {0, 1}, interpreted as alpha/(alpha+beta) x Z
    Output: {0,1}
"""

class PolicyNetwork:
    def __init__(self, state_dim,
                 actor_hidden_dim):
        """
        Initialize the policy network.
        :param state_dim:                   Dimensionality of the state space
        :param actor_hidden_dim:            A list of positive integers
        """
        self.state_dim = state_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.policy_model = self._set_policy_model()

    def _set_policy_model(self):
        """
        Initialize the policy network
        :return: A policy model
        """
        inputs = keras.layers.Input(shape=(self.state_dim,),
                                    name='policy_input_state')
        x = inputs
        for ii in range(len(self.actor_hidden_dim)):
            x = keras.layers.Dense(self.actor_hidden_dim[ii],
                                   activation=tf.nn.relu,
                                   kernel_initializer=initializers.RandomUniform(minval=-.1, maxval=.1, seed=None),
                                   name='policy_h%d' % ii)(x)
        x = keras.layers.Dense(2,
                               activation=tf.nn.softmax,
                               kernel_initializer=initializers.RandomUniform(minval=-.1, maxval=.1, seed=None),
                               name='policy_output')(x)
        policy_model = keras.models.Model(inputs=inputs, outputs=x)
        return policy_model
    
    def get_action(self, s):
        """
        Evaluate self.policy_model and get the action predictions corresponding to s
        :param s: A batch of states
        :return: A batch of action predictions (in {0, 1})
        """
        # print(self.policy_model.predict(s))
        # print("get_action("+str(s)+") = "+str(np.argmax(self.policy_model.predict(s))))
        
        score = self.policy_model.predict(s)
        score[:,1] = np.random.rand(score.shape[0])
        return np.argmax(score, axis=1)
        # return np.argmax(self.policy_model.predict(s), axis=1)
    
    def get_nn_weights(self):
        """
        Get all weights of self.policy_model
        :return: return a list of all parameters of one policy network
        """
        weights = []
        for layer in self.policy_model.layers:
            layer_weights = layer.get_weights()
            if len(weights) == 0:
                weights = [layer_weights]
            else:
                weights.append(layer_weights)
        return weights
    
    def set_nn_weights(self, weights):
        """
        Set the parameters in a policy network
        :param weights: a list of parameters
        :return: None
        """
        for w, layer in zip(weights, self.policy_model.layers):
            # print(layer.get_weights())
            # print(w)
            # print('~~~')
            layer.set_weights(w)
    
    def get_nn_weight_sum(self, weights):
        s = 0
        for w_layer in weights:
            if len(w_layer) > 0:
                for w in w_layer:
                    s += np.sum(np.abs(w))
        return s