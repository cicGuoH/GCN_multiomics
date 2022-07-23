# -*- coding: utf-8 -*-
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from datetime import datetime
from GCN.utils import normalize_adj
from GCN.models import BaseGCNModel, BaseGCNModel_addSE, BaseGCNModel_addSE_attn
from GCN.layers import MatthewsCorrelationCoefficient as MCC
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


data_file_path = "../Data/brca_pam50_hvG3167.hdf5"

with h5py.File(data_file_path,"r") as f:
    
    train_data = f["Tr_dataset"]["tr_data"][:,:,:].astype(np.float32)
    train_y = f["Tr_dataset"]["tr_pam50"]["PAM50"][:].astype(np.int32)
    test_data = f["Te_dataset"]["te_data"][:,:,:].astype(np.float32)
    test_y = f["Te_dataset"]["te_pam50"]["PAM50"][:].astype(np.int32)
    
    graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]


train_X = tf.convert_to_tensor(train_data, dtype=tf.float32)
test_X = tf.convert_to_tensor(test_data, dtype=tf.float32)
train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)

train_y_onehot = tf.one_hot(train_y, depth=5, on_value=1, off_value=0)
test_y_onehot = tf.one_hot(test_y, depth=5, on_value=1, off_value=0)

graph_norm = normalize_adj(graph_data.astype(np.float32), False)
graph_norm_dense = tf.sparse.to_dense(graph_norm)

# model = BaseGCNModel(gcn_hid_dim=32,
#                       fc_dim_1=1024, fc_dim_2=512, output_dim=5,
#                       init_graph=graph_norm_dense,
#                       dropout_rate=0.8,
#                       pool_method="max")

# model = BaseGCNModel_addSE(gcn_hid_dim=32,
#                             se_embed_dim=32, se_input_dim=train_data.shape[-1],  
#                             fc_dim_1=1024, fc_dim_2=512, output_dim=5, 
#                             init_graph=graph_norm_dense,
#                             dropout_rate=0.5,
#                             pool_method="max")

model = BaseGCNModel_addSE_attn(gcn_hid_dim=32,
                                se_embed_dim=32, se_input_dim=train_data.shape[-1],  
                                fc_dim_1=512, fc_dim_2=256, output_dim=5,
                                init_graph=graph_norm_dense,
                                dropout_rate=0.6,
                                pool_method="max")


initial_learning_rate = 5*1e-3
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                              decay_steps=30,
                                                              decay_rate=0.96,
                                                              staircase=True)

loss_tracker = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
metrics_auc = tf.keras.metrics.AUC(from_logits=False)
metrics_acc = tf.keras.metrics.CategoricalAccuracy()
metrics_mcc = MCC(num_classes=5, name="mcc")
tb_callback = tf.keras.callbacks.TensorBoard("./training_logs/logs_BaseGCN_addSE_attn_0515", update_freq="batch", histogram_freq=1)
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(min_delta=1e-4, 
                                                          patience=50, 
                                                          restore_best_weights=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./modelSave/BaseGCN_addSE_attn_0515.ckpt", 
                                                 save_weights_only=True, 
                                                 verbose=1,
                                                 save_best_only=True)

model.compile(optimizer=optimizer, loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])
model.fit(train_X, train_y_onehot, 
          batch_size=64, 
          epochs=800,
          callbacks=[tb_callback, earlyStopping_callback, cp_callback],
          validation_data=[test_X, test_y_onehot],
          shuffle=True, 
          )


# load weight and evalute
# model = BaseGCNModel(gcn_hid_dim=32,
#                       fc_dim_1=1024, fc_dim_2=512, output_dim=5,
#                       init_graph=graph_norm_dense,
#                       dropout_rate=0.8,
#                       pool_method="max")

# model = BaseGCNModel_addSE(gcn_hid_dim=32,
#                             se_embed_dim=32, se_input_dim=train_data.shape[-1],  
#                             fc_dim_1=1024, fc_dim_2=512, output_dim=5, 
#                             init_graph=graph_norm_dense,
#                             dropout_rate=0.5,
#                             pool_method="max")

model.load_weights("./modelSave/BaseGCN_addSE_attn_0515.ckpt")
model.compile(optimizer=optimizer, loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])
model.evaluate(train_X, train_y_onehot)
model.evaluate(test_X, test_y_onehot)

#get SE attention score
model.SE(train_X)

