# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import optimizers
from tensorflow.keras import Model, layers
from GCN.layers import Dense, GraphConvolution, DiffPool, GraphPool, MultiHeadAttention, GraphAttention
import scipy.sparse as sp
from GCN.utils import normalize_adj
import numpy as np 

class SE_module(Model):
    def __init__(self, embed_dim, output_dim, dropout_rate=0., pool_method="max"):
        super().__init__()
        self.globalPool = GraphPool(method=pool_method, axis=1)
        self.fc_1 = Dense(embed_dim, dropout_rate=dropout_rate, activation="relu", name="fc_1")
        self.fc_2 = Dense(embed_dim, dropout_rate=dropout_rate, activation="relu", name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="sigmoid", name="op")
    
    def call(self, inputs, training=True):
        x = inputs #N * F * C
        # global pool
        x = self.globalPool(x) #N * C
        x = self.fc_1(x) # N * C1
        x = self.fc_2(x) # N * C1
        x = self.op(x) # N * C
        return x
    
class BaseMLPModel(Model):
    
    def __init__(self, 
                 attn_embed_dim, attn_out_dim,  
                 fc_dim_1, fc_dim_2, output_dim,
                 num_heads=4, 
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=attn_embed_dim, out_dim=attn_out_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), dropout_rate=dropout_rate)
        self.globalPool = GraphPool(method=pool_method, axis=1)
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        
    def call(self, inputs, training=True):
        x = inputs
        # multi_head_attn
        x_t = tf.transpose(x, [0,2,1])
        x_t_attn = self.attn(x_t, mask=None)
        x += tf.transpose(x_t_attn,[0,2,1])
        x = self.bn_1(x, training=training)
        x = self.globalPool(x)
        x = self.fc_1(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.bn_3(x, training=training)
        logits = self.op(x)
        return logits
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class BaseGCNModel(Model):
    
    def __init__(self, 
                 gcn_hid_dim, 
                 fc_dim_1, fc_dim_2, output_dim,
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        self.graph = init_graph
        self.globalPool = GraphPool(method=pool_method, axis=1)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation="relu", name="gcn")
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        
    def call(self, inputs, training=True):
        x = inputs
        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        x = self.globalPool(x)
        
        x = self.fc_1(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.bn_3(x, training=training)
        logits = self.op(x)
        return logits
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits, _ = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

class BaseGCNModel_singleOmics(Model):
    def __init__(self, 
                 gcn_hid_dim, 
                 fc_dim_1, fc_dim_2, output_dim,
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        self.graph = init_graph
        self.globalPool = GraphPool(method=pool_method, axis=1)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation="relu", name="gcn")
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        
    def call(self, inputs, training=True):
        rank = len(inputs.shape)
        if rank < 3:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        x = self.globalPool(x)
        x = self.fc_1(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.bn_3(x, training=training)
        logits = self.op(x)
        return logits
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
class BaseGCNModel_addSE(Model):
    def __init__(self, 
                 gcn_hid_dim, 
                 se_embed_dim, se_input_dim,  
                 fc_dim_1, fc_dim_2, output_dim,
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        
        self.graph = init_graph
        self.SE = SE_module(embed_dim=se_embed_dim, output_dim=se_input_dim)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="gcn")
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        self.globalPool = GraphPool(method=pool_method)
        
    def call(self, inputs, training=True):
        x = inputs
        # SE channel attention
        attn_score = tf.expand_dims(self.SE(x), axis=1)
        x = x * attn_score
        x += inputs
        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        x = self.globalPool(x)
        x = self.fc_1(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.bn_3(x, training=training)
        logits = self.op(x)
        return logits
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

# Now used
class BaseGCNModel_addAttn(Model):
    def __init__(self, 
                 gcn_hid_dim, 
                 attn_embed_dim, attn_out_dim, 
                 fc_dim_1, fc_dim_2, output_dim,
                 num_heads=4, 
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        
        self.graph = init_graph
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=attn_embed_dim, out_dim=attn_out_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001), dropout_rate=dropout_rate)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="gcn")
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation=None, name="output")
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        self.globalPool = GraphPool(method=pool_method)
        
    def call(self, inputs, training=True):
        activations = []
        x = inputs
        # multi_head_attn
        x_t = tf.transpose(x, [0,2,1])
        x_t_attn, _ = self.attn(x_t, mask=None)
        x += tf.transpose(x_t_attn,[0,2,1])
        # gcn
        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        x = self.globalPool(x)
        activations.append(x)
        x = self.fc_1(x, training=training)
        activations.append(x)
        x = self.bn_2(x, training=training)
        x = self.fc_2(x, training=training)
        activations.append(x)
        x = self.bn_3(x, training=training)
        acts = self.op(x)
        activations.append(acts)
        logits = tf.nn.softmax(acts)
        return logits, activations
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, _ = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits, _ = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

# not used
class BaseGCNModel_addSE_attn(Model):
    def __init__(self, 
                 gcn_hid_dim, 
                 se_embed_dim, se_input_dim,  
                 # fc_dim_1, fc_dim_2, output_dim,
                 fc_dim_1, output_dim,
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        
        self.graph = init_graph
        self.SE = SE_module(embed_dim=se_embed_dim, output_dim=se_input_dim)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="gcn")
        self.multi_head_attn = MultiHeadAttention(num_heads=4, key_dim=gcn_hid_dim, dropout_rate=dropout_rate)
        self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        # self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_2")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        # self.bn_4 = layers.BatchNormalization(axis=1)
        self.globalPool = GraphPool(method=pool_method)
        
    def call(self, inputs, training=True):
        x = inputs
        # SE channel attention
        attn_score = tf.expand_dims(self.SE(x), axis=1)
        x = x * attn_score
        x += inputs
        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        # get mask
        mask = tf.expand_dims(tf.cast(self.graph > 0., tf.float32), axis=0)
        # multi-head attention
        x_attn, _ = self.multi_head_attn(x, mask=mask)
        x += x_attn
        x = self.bn_2(x, training=training)
        
        x = self.globalPool(x)
        x = self.fc_1(x, training=training)
        x = self.bn_3(x, training=training)
        # x = self.fc_2(x, training=training)
        # x = self.bn_4(x)
        logits = self.op(x)
        return logits
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}



class BaseGCNModel_add_CRattn(Model):
    def __init__(self, 
                 gcn_hid_dim, 
                 attn_embed_dim_1, attn_input_dim_1, num_heads_1, 
                 attn_embed_dim_2, attn_input_dim_2, num_heads_2,  
                 fc_dim, output_dim,
                 init_graph=None,
                 dropout_rate=0.,
                 pool_method="max",
                 **kwargs):
        
        super().__init__()
        
        self.graph = init_graph
        self.col_attn = MultiHeadAttention(num_heads=num_heads_1, key_dim=attn_embed_dim_1, out_dim=attn_input_dim_1, kernel_regularizer=tf.keras.regularizers.l2(0.001), dropout_rate=dropout_rate)
        self.gcn = GraphConvolution(gcn_hid_dim, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="gcn")
        self.row_attn = MultiHeadAttention(num_heads=num_heads_2, key_dim=attn_embed_dim_2, out_dim=attn_input_dim_2, kernel_regularizer=tf.keras.regularizers.l2(0.001), dropout_rate=dropout_rate)
        self.fc = Dense(fc_dim, dropout_rate=dropout_rate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="fc_1")
        self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        self.bn_1 = layers.BatchNormalization(axis=1)
        self.bn_2 = layers.BatchNormalization(axis=1)
        self.bn_3 = layers.BatchNormalization(axis=1)
        self.globalPool = GraphPool(method=pool_method)

    def call(self, inputs, training=True):
        x = inputs
        
        # multi_head_attn
        x_t = tf.transpose(x, [0,2,1])
        x_t_attn, col_attn_score = self.col_attn(x_t, mask=None)
        x += tf.transpose(x_t_attn,[0,2,1])

        x = self.gcn(x, self.graph, training=training)
        x = self.bn_1(x, training=training)
        # get mask
        mask = tf.expand_dims(tf.cast(self.graph > 0., tf.float32), axis=0)
        # multi-head attention
        x_attn, row_attn_score = self.row_attn(x, mask=mask)
        x += x_attn
        x = self.bn_2(x, training=training)
        
        x = self.globalPool(x)
        x = self.fc(x, training=training)
        x = self.bn_3(x, training=training)
        logits = self.op(x)
        return logits, col_attn_score, row_attn_score
    
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits,_,_, = self(x, training=True)
            #compute loss
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        #compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        #update_weight
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #compute metrics
        self.compiled_metrics.update_state(y, logits)
        return {m.name:m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        logits,_,_ = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, logits, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, logits)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

# not used
# class BaseGATModel_addSE(Model):
    
#     def __init__(self, 
#                  gat_hid_dim, 
#                  se_embed_dim, se_input_dim,  
#                  fc_dim_1, fc_dim_2, output_dim,
#                  init_graph=None,
#                  dropout_rate=0.,
#                  pool_method="max",
#                  is_normalized=True,
#                  add_graph_loss=False,
#                  **kwargs):
        
#         super().__init__()
#         self.SE = SE_module(embed_dim=se_embed_dim, output_dim=se_input_dim)
#         self.gat = GraphAttention(F_=gat_hid_dim, attn_heads=4, attn_heads_reduction="average", name="gat")
#         self.graph = init_graph
#         self.fc_1 = Dense(fc_dim_1, dropout_rate=dropout_rate, activation="relu", name="fc_1")
#         self.fc_2 = Dense(fc_dim_2, dropout_rate=dropout_rate, activation="relu", name="fc_2")
#         self.op = Dense(output_dim, dropout_rate=dropout_rate, activation="softmax", name="output")
        
#         self.bn_1 = layers.BatchNormalization(axis=1)
#         self.bn_2 = layers.BatchNormalization(axis=1)
#         self.globalPool = GraphPool(method=pool_method)
#         self.is_normalized = is_normalized
#         self.add_graph_loss = add_graph_loss
        
#     def call(self, inputs, training=True):
#         x = self.gat(inputs, self.graph)
#         # x2, _, = self.gpool1(x1, self.graph)
#         x = self.bn_1(x, training=training)
#         x = self.globalPool(x)
#         x = self.fc_1(x, training=training)
#         x = self.fc_2(x, training=training)
#         logits = self.op(x)
#         return logits
    
#     @tf.function
#     def train_step(self, data):
#         x, y = data
#         with tf.GradientTape() as tape:
#             logits = self(x, training=True)
#             #compute loss
#             loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
#         #compute gradients
#         grads = tape.gradient(loss, self.trainable_variables)
#         #update_weight
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#         #compute metrics
#         self.compiled_metrics.update_state(y, logits)
#         return {m.name:m.result() for m in self.metrics}
    
#     @tf.function
#     def test_step(self, data):
#         # Unpack the data
#         x, y = data
#         # Compute predictions
#         logits = self(x, training=False)
#         # Updates the metrics tracking the loss
#         self.compiled_loss(y, logits, regularization_losses=self.losses)
#         # Update the metrics.
#         self.compiled_metrics.update_state(y, logits)
#         # Return a dict mapping metric names to current value.
#         # Note that it will include the loss (tracked in self.metrics).
#         return {m.name: m.result() for m in self.metrics}



if __name__ == "__main__":
    
    import h5py
    data_file_path = "../Data/brca_pam50.hdf5"

    with h5py.File(data_file_path,"r") as f:
        
        train_data = f["Tr_dataset"]["tr_data"][:,:,:].astype(np.float32)
        train_y = f["Tr_dataset"]["tr_pam50"]["PAM50"][:].astype(np.float32)
        test_data = f["Te_dataset"]["te_data"][:,:,:].astype(np.float32)
        test_y = f["Te_dataset"]["te_pam50"]["PAM50"][:].astype(np.float32)
        
        graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]
   
    graph_norm = normalize_adj(graph_data, False)
    model = BaseGCNModel(gcn_hid_dim_1=16, 
                         fc_dim_1=256, fc_dim_2=128, output_dim=5,
                         cluster_dim_1=1000, cluster_dim_2=512, 
                         )
