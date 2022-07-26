# -*- coding: utf-8 -*-
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from GCN.utils import normalize_adj
from GCN.models import BaseGCNModel, BaseGCNModel_addSE, BaseGCNModel_addSE_attn, BaseGCNModel_addAttn, BaseMLPModel, BaseGCNModel_add_CRattn
from GCN.layers import MatthewsCorrelationCoefficient as MCC
import pandas as pd
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

data_file_path = "../Data/brca_pam50_hvG3167_5hop_5cv.hdf5"

result_df = pd.DataFrame()
for num_fold in range(1,6): 
    with h5py.File(data_file_path,"r") as f:
        train_data = f["fold_%d"%num_fold]["train_data"][:,:,:].astype(np.float32)
        train_y = f["fold_%d"%num_fold]["train_label"][:].astype(np.int32)
        test_data = f["fold_%d"%num_fold]["test_data"][:,:,:].astype(np.float32)
        test_y = f["fold_%d"%num_fold]["test_label"][:].astype(np.int32)
        graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]
    
    
    train_X = tf.convert_to_tensor(train_data, dtype=tf.float32)
    test_X = tf.convert_to_tensor(test_data, dtype=tf.float32)
    train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
    
    train_y_onehot = tf.one_hot(train_y, depth=5, on_value=1, off_value=0)
    test_y_onehot = tf.one_hot(test_y, depth=5, on_value=1, off_value=0)
    
    graph_norm = normalize_adj(graph_data.astype(np.float32), False)
    graph_norm_dense = tf.sparse.to_dense(graph_norm)
    
    # model = BaseGCNModel(gcn_hid_dim=64,
    #                       fc_dim_1=1024, fc_dim_2=512, output_dim=5,
    #                       init_graph=graph_norm_dense,
    #                       dropout_rate=0.6,
    #                       pool_method="max")
    
    # model = BaseGCNModel_addSE(gcn_hid_dim=32,
    #                             se_embed_dim=32, se_input_dim=train_data.shape[-1],  
    #                             fc_dim_1=512, fc_dim_2=256, output_dim=5, 
    #                             init_graph=graph_norm_dense,
    #                             dropout_rate=0.5,
    #                             pool_method="max")
    model = BaseGCNModel_addAttn(gcn_hid_dim=32,
                                  attn_embed_dim=256, attn_out_dim=train_data.shape[1], 
                                  fc_dim_1=512, fc_dim_2=256, output_dim=5, 
                                  num_heads=7, 
                                  init_graph=graph_norm_dense,
                                  dropout_rate=0.5,
                                  pool_method="max")
    
    # model = BaseMLPModel(attn_embed_dim=256, attn_out_dim=train_data.shape[1], 
    #                       fc_dim_1=512, fc_dim_2=256, output_dim=5, 
    #                       num_heads=4,
    #                       dropout_rate=0.8,
    #                       pool_method="max")
    
    # model = BaseGCNModel_add_CRattn(gcn_hid_dim=32, 
    #                                 attn_embed_dim_1=256, attn_input_dim_1=train_data.shape[1], num_heads_1=7, 
    #                                 attn_embed_dim_2=64, attn_input_dim_2=32, num_heads_2=4,  
    #                                 fc_dim=512, output_dim=5,
    #                                 init_graph=graph_norm_dense,
    #                                 pool_method="max",
    #                                 dropout_rate=0.5)
    
    initial_learning_rate = 1e-3
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                  decay_steps=30,
                                                                  decay_rate=0.96,
                                                                  staircase=True)
    
    loss_tracker = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    metrics_auc = tf.keras.metrics.AUC(from_logits=False)
    metrics_acc = tf.keras.metrics.CategoricalAccuracy()
    metrics_mcc = MCC(num_classes=5, name="mcc")
    tb_callback = tf.keras.callbacks.TensorBoard(r".\\training_logs\\MDC_omics_5hop\\logs_BaseGCN_Attn_h7_fold{}".format(num_fold), update_freq="batch", histogram_freq=1)
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(min_delta=1e-4, 
                                                              monitor='val_categorical_accuracy',
                                                              patience=50, 
                                                              restore_best_weights=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=".\\modelSave\\MDC_omics_5hop\\BaseGCN_Attn_h7_fold{}\\ckpt".format(num_fold), 
                                                     monitor='val_categorical_accuracy',
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
    res = model.evaluate(test_X, test_y_onehot)
    result_df["fold_%d"%num_fold] = res
    
result_df.index = ["Loss", "AUC", "ACC", "MCC"]
print(result_df.mean(axis=1))
print(result_df.std(axis=1))

