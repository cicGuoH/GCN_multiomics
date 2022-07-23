# -*- coding: utf-8 -*-
from tensorflow.python.ops import gen_nn_ops
import numpy as np
import tensorflow as tf
import scipy
import time

class LayerRelevancePropagation:
    def __init__(self, weights, activations):
        self.epsilon = 1e-10
        self.weights = weights
        self.activations = activations
    def relevance_propagation(self):
        relevance = self.activations[-1]
        for i in range(1, len(self.activations)):
            relevance = self.lrp_dense(self.activations[-(i+1)], self.weights[-i], relevance)
        return relevance
    def lrp_dense(self, x, w, r):
        """
        z_plus
        """
        w_pos = tf.maximum(0., w)
        z = tf.matmul(x, w_pos) + self.epsilon
        s = r/z
        c = tf.matmul(s, tf.transpose(w_pos, [1,0]))
        return c * x
