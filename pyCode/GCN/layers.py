# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from GCN.utils import normalize_adj
from tensorflow.keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
from keras import activations, constraints, initializers, regularizers

import numpy as np

class Dense(Layer):
    """
    Dense Layer: full connected layer
    """
    
    def __init__(self, 
                 output_dim, 
                 dropout_rate=0., 
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation="relu", 
                 bias=False, 
                 logging=True, 
                 **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.bias = bias
        self.vars = {}
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        self.logging = logging
        
        if self.logging:
            self._log_vars()
    
    def build(self, input_shape):
        self.vars["kernel"] = self.add_weight(name=self.name+"_kernel", 
                                              shape=[input_shape[-1], self.output_dim],
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              trainable=True)
        
        if self.bias:
            self.vars["bias"] =self.add_weight(name=self.name+"_bias",
                                               shape=[self.output_dim,],
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               trainable=True) 
        self.build = True
        
    def call(self, inputs, training=True):
        
        rank = inputs.shape.rank
        x = inputs
        # dropout
        if self.dropout_rate > 0. and training:
            x = tf.nn.dropout(x, 1-self.dropout_rate)
        
        # transform
        if rank == 2:
            x_z = tf.matmul(x, self.vars["kernel"])
        else:
            x_z = tf.tensordot(x, self.vars["kernel"], [[rank - 1], [0]])
        # add bias
        if self.bias:
            x_z += self.vars["bias"]
    
        if self.activation != None:
            output = self.activation(x_z)
        else:
            output = x_z
        return output
    
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    def __init__(self, 
                 output_dim, 
                 dropout_rate=0., 
                 kernel_initializer="glorot_uniform",
                 bias_initializer="Zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation="relu", 
                 bias=False, 
                 logging=True,
                 **kwargs):
        super().__init__()
        
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.bias = bias
        self.logging = logging
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        self.vars = {}
        if self.logging:
            self._log_vars()
        
    def build(self, input_shape):
        self.vars["kernel"] = self.add_weight(name=self.name+"kernel", 
                                              shape=[input_shape[-1], self.output_dim],
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              trainable=True)
        
        if self.bias:
            self.vars["bias"] =self.add_weight(name=self.name+"bias", 
                                               shape=[self.output_dim,],
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               trainable=True) 
        self.build = True
        
    def call(self, inputs, adj, is_batch=False, training=True):
        
        x = inputs
        b_size = tf.shape(x)[0]
        
        # dropout
        if self.dropout_rate > 0 and training:
            x = tf.nn.dropout(x, 1-self.dropout_rate)
            
        # convolve (D^-1/2(A+I)D^-1/2Wx)
        x_pre = tf.einsum("nij,jk->nik",x, self.vars["kernel"]) 
        # print(x_pre.shape)
        if is_batch:
            if len(tf.shape(adj)) != 3:
                adj = tf.tile(tf.expand_dims(adj, axis=0), [b_size, 1, 1])
            x_z = tf.einsum("nij,njk->nik", adj, x_pre)
            
        else:
            x_z = tf.einsum("ij,njk->nik", adj, x_pre)
        if self.bias:
            x_z += self.vars["bias"]
        if self.activation != None:
            output = self.activation(x_z)
        else:
            output = x_z
        return output
    
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class MultiHeadAttention(Layer):
    def __init__(self,
                 num_heads,
                 key_dim, out_dim, 
                 dropout_rate=0.,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs
                 ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.vars = {}
        
    def build(self, input_shape):
        
        self.vars["Q_kernel"] = self.add_weight(name=self.name+"_Q_kernel",
                                                shape=[input_shape[-1], self.num_heads * self.key_dim],
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True)
        self.vars["K_kernel"] = self.add_weight(name=self.name+"_K_kernel",
                                                shape=[input_shape[-1], self.num_heads * self.key_dim],
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True)
        self.vars["V_kernel"] = self.add_weight(name=self.name+"_V_kernel",
                                                shape=[input_shape[-1], self.num_heads * self.key_dim],
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True)
        self.vars["W"] = self.add_weight(name=self.name+"_W",
                                                shape=[self.num_heads * self.key_dim, self.out_dim],
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                trainable=True)
        

    def call(self, inputs, mask=None, training=True):
        Q_ = tf.einsum("nij,jk->nik", inputs, self.vars["Q_kernel"])
        K_ = tf.einsum("nij,jk->nik", inputs, self.vars["K_kernel"])
        V_ = tf.einsum("nij,jk->nik", inputs, self.vars["V_kernel"])
        
        Q_multi_head = tf.concat(tf.split(Q_, self.num_heads, axis=-1), axis=0) #n*h,m,f
        K_multi_head = tf.concat(tf.split(K_, self.num_heads, axis=-1), axis=0) 
        V_multi_head = tf.concat(tf.split(V_, self.num_heads, axis=-1), axis=0)
            
        # matual
        attn = tf.einsum("nmi,nki->nmk", Q_multi_head, K_multi_head) #n*h,m,m
        if mask is not None:
            attn = attn * mask + -1e9 * (1 - mask)
        scaled_attn = tf.nn.softmax(attn / (Q_.shape[-1])**0.5, axis=-1)
        # dropout
        if self.dropout_rate > 0. and training:
            scaled_attn = tf.nn.dropout(scaled_attn, 1-self.dropout_rate)
        
        # outputs
        outputs = tf.matmul(scaled_attn, V_multi_head) #n*h,m,f
        # reshape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=-1) # n,m,f*h
        # linear
        outputs = tf.einsum("nij,jk->nik", outputs, self.vars["W"])
        return outputs, scaled_attn
    
    def get_attn_score(self, inputs, mask=None, training=True):
        Q_ = tf.einsum("nij,jk->nik", inputs, self.vars["Q_kernel"])
        K_ = tf.einsum("nij,jk->nik", inputs, self.vars["K_kernel"])
        Q_multi_head = tf.concat(tf.split(Q_, self.num_heads, axis=-1), axis=0) #n*h,m,f
        K_multi_head = tf.concat(tf.split(K_, self.num_heads, axis=-1), axis=0) 
            
        # matual
        attn = tf.einsum("nmi,nki->nmk", Q_multi_head, K_multi_head) #n*h,m,m
        if mask is not None:
            attn = attn * mask + -1e9 * (1 - mask)
        scaled_attn = tf.nn.softmax(attn / (Q_.shape[-1])**0.5, axis=-1)
        attn_score = tf.split(scaled_attn, self.num_heads, axis=0)
        return attn_score 

# Not used now
#reference to https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, adj):
        X = inputs  # Node features (S x N x F)
        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = tf.einsum("nij,jk->nik", X, kernel) # (S x N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = tf.einsum("nij,jk->nik", features, attention_kernel[0])    # (S x N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = tf.einsum("nij,jk->nik", features, attention_kernel[1])  # (S x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs, [0,2,1])  # (S x N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - tf.expand_dims(tf.cast(adj > 0., tf.float32), axis=0))
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense, axis=-1)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(1-self.dropout_rate)(dense)  # (S x N x N)
            dropout_feat = Dropout(1-self.dropout_rate)(features)  # (S x N x F')

            # Linear combination with neighbors' features
            node_features = tf.einsum("nij,njk->nik", dropout_attn, dropout_feat)  # (S x N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

# Not used
class DiffPool(Layer):
    """
    reference to torch_geometric.nn.dense.diff_pool
    """
    def __init__(self, cluster_dim, eps=1e-3, add_graph_loss=False, **kwargs):
        super().__init__()
        self.eps = 1e-3
        self.assignment_layer = GraphConvolution(cluster_dim, name="assignment")
        self.add_graph_loss = add_graph_loss
        
    def call(self, inputs, adj, training=True):
        
        x = tf.expand_dims(inputs, 0) if len(inputs.shape)==2 else inputs #N * F * Ne
        b_size = tf.shape(x)[0]
        # adj -> adj_list
        if len(tf.shape(adj)) != 3:
            adj = tf.tile(tf.expand_dims(adj, axis=0), [b_size, 1, 1])
        s = tf.nn.softmax(self.assignment_layer(x, adj, is_batch=True, training=training), axis=-1) # N * F * Nc
        out_fea = tf.einsum("ncf,nfe->nce", tf.transpose(s,[0,2,1]), x) # N * Nc * Ne
        if self.add_graph_loss:
            # adj * s
            adj_s_mul = tf.einsum("nij,njk->nik", adj, s)
            out_adj = tf.einsum("nij,nik->njk", s, adj_s_mul) #n*f*c,n*f*c-> n*c*c
            recons_adj = tf.einsum("nij,nkj->nik", s, s)
            
            link_loss = tf.norm((adj-recons_adj), axis=[-2,-1])
            ent_loss = tf.reduce_mean(tf.reduce_sum((- s * tf.math.log(s+self.eps)), axis=-1), axis=-1) #one node one cluster
            self.graph_link_loss = link_loss
            self.graph_ent_loss = ent_loss
        else:
            # adj * s
            adj_s_mul = tf.einsum("nij,njk->nik", adj, s)
            out_adj = tf.einsum("nij,nik->njk", s, adj_s_mul) #n*f*c,n*f*c-> n*c*c
        return out_fea, out_adj

class GraphPool(Layer):
    
    def __init__(self, method, axis=-1, **kwargs):
        super().__init__()
        self.method = method
        self.axis = axis
    
    def call(self, inputs):
        if self.method == "max":
            res = tf.reduce_max(inputs, axis=self.axis, keepdims=False)
        
        elif self.method == "mean":
            res = tf.reduce_mean(inputs, axis=self.axis, keepdims=False)
        
        elif self.method == "add":
            res = tf.reduce_sum(inputs, axis=self.axis, keepdims=False)
            
        else:
            print("Method '%s' is not defined" % self.method)
        return res


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(
        self,
        num_classes,
        name="MatthewsCorrelationCoefficient",
        **kwargs,
    ):
        """Creates a Matthews Correlation Coefficient instance."""
        super().__init__(name=name)
        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        new_conf_mtx = tf.math.confusion_matrix(
            labels=tf.argmax(y_true, 1),
            predictions=tf.argmax(y_pred, 1),
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=self.dtype,
        )

        self.conf_mtx.assign_add(new_conf_mtx)

    def result(self):

        true_sum = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_sum = tf.reduce_sum(self.conf_mtx, axis=0)
        num_correct = tf.linalg.trace(self.conf_mtx)
        num_samples = tf.reduce_sum(pred_sum)

        # covariance true-pred
        cov_ytyp = num_correct * num_samples - tf.tensordot(true_sum, pred_sum, axes=1)
        # covariance pred-pred
        cov_ypyp = num_samples ** 2 - tf.tensordot(pred_sum, pred_sum, axes=1)
        # covariance true-true
        cov_ytyt = num_samples ** 2 - tf.tensordot(true_sum, true_sum, axes=1)

        mcc = cov_ytyp / tf.math.sqrt(cov_ytyt * cov_ypyp)

        if tf.math.is_nan(mcc):
            mcc = tf.constant(0, dtype=self.dtype)

        return mcc

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v,
                np.zeros((self.num_classes, self.num_classes), v.dtype.as_numpy_dtype),
            )

    def reset_states(self):
        return self.reset_state()
    