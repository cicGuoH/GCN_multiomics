# -*- coding: utf-8 -*-
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from GCN.utils import normalize_adj
from GCN.models import BaseGCNModel, BaseGCNModel_addSE, BaseGCNModel_addAttn, BaseGCNModel_add_CRattn
from GCN.layers import MatthewsCorrelationCoefficient as MCC
from sklearn.metrics import confusion_matrix
import pandas as pd
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from GCN.lrp import LayerRelevancePropagation

data_file_path = "../Data/brca_pam50_hvG3167_5cv.hdf5"
patientID_cv = pd.read_csv("../patientID_CV.csv")

#%% GCN + SE
result_ls = []
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
    #                      fc_dim_1=1024, fc_dim_2=512, output_dim=5,
    #                      init_graph=graph_norm_dense,
    #                      dropout_rate=0.6,
    #                      pool_method="max")
    
    model = BaseGCNModel_addSE(gcn_hid_dim=32,
                                se_embed_dim=32, se_input_dim=train_data.shape[-1],  
                                fc_dim_1=512, fc_dim_2=256, output_dim=5, 
                                init_graph=graph_norm_dense,
                                dropout_rate=0.5,
                                pool_method="max")
    
    # model = BaseGCNModel_addAttn_v2(gcn_hid_dim=32,
    #                                 attn_embed_dim=256, attn_out_dim=train_data.shape[1], 
    #                                 fc_dim_1=512, fc_dim_2=256, output_dim=5, 
    #                                 init_graph=graph_norm_dense,
    #                                 dropout_rate=0.5,
    #                                 pool_method="max")
    
    # model = BaseGCNModel_addAttn(gcn_hid_dim=32,
    #                             attn_embed_dim=256,  
    #                             fc_dim_1=512, fc_dim_2=256, output_dim=5, 
    #                             init_graph=graph_norm_dense,
    #                             dropout_rate=0.5,
    #                             pool_method="max")
    
    # model = BaseGCNModel_addSE_attn(gcn_hid_dim=32,
    #                                 se_embed_dim=32, se_input_dim=train_data.shape[-1],  
    #                                 fc_dim_1=512, output_dim=5,
    #                                 init_graph=graph_norm_dense,
    #                                 dropout_rate=0.6,
    #                                 pool_method="max")
    
    
    initial_learning_rate = 5*1e-3
    loss_tracker = tf.keras.losses.CategoricalCrossentropy()
    metrics_auc = tf.keras.metrics.AUC(from_logits=False)
    metrics_acc = tf.keras.metrics.CategoricalAccuracy()
    metrics_mcc = MCC(num_classes=5, name="mcc")
    # load weight and evalute
    model.load_weights("./modelSave/BaseGCN_addSE_fold%d\\ckpt"%num_fold)
    model.compile(loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])
    
    #get SE attention score
    train_se_score = model.SE(train_X)
    test_se_score = model.SE(test_X)
    all_se_score = pd.concat([pd.DataFrame(train_se_score.numpy()), pd.DataFrame(test_se_score.numpy())], axis=0)
    all_se_score.columns = ["C1","C2","C3"]
    all_se_score.index = patientID_cv["fold_%d"%num_fold]
    result_ls.append(all_se_score)
result_df = pd.concat(result_ls, axis=1)
result_df.to_csv("../GCN_SE_score_cv.csv")

#%% GCN+col_Attn
cm_dfs = []
for num_fold in range(1,6):
    num_fold = 1
    with h5py.File(data_file_path,"r") as f:
        train_data = f["fold_%d"%num_fold]["train_data"][:,:,:].astype(np.float32)
        train_y = f["fold_%d"%num_fold]["train_label"][:].astype(np.int32)
        test_data = f["fold_%d"%num_fold]["test_data"][:,:,:].astype(np.float32)
        test_y = f["fold_%d"%num_fold]["test_label"][:].astype(np.int32)
        graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]
        nodes_id = f["PPI_network"]["Nodes"][:].astype("U")
    
    train_X = tf.convert_to_tensor(train_data, dtype=tf.float32)
    test_X = tf.convert_to_tensor(test_data, dtype=tf.float32)
    train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
    
    train_y_onehot = tf.one_hot(train_y, depth=5, on_value=1, off_value=0)
    test_y_onehot = tf.one_hot(test_y, depth=5, on_value=1, off_value=0)
    
    graph_norm = normalize_adj(graph_data.astype(np.float32), False)
    graph_norm_dense = tf.sparse.to_dense(graph_norm)
    
    model = BaseGCNModel_addAttn(gcn_hid_dim=32,
                                 attn_embed_dim=256, attn_out_dim=train_data.shape[1], 
                                 num_heads=7,
                                 fc_dim_1=512, fc_dim_2=256, output_dim=5, 
                                 init_graph=graph_norm_dense,
                                 dropout_rate=0.5,
                                 pool_method="max")
    
    initial_learning_rate = 5*1e-3
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                  decay_steps=30,
                                                                  decay_rate=0.96,
                                                                  staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss_tracker = tf.keras.losses.CategoricalCrossentropy()
    metrics_auc = tf.keras.metrics.AUC(from_logits=False)
    metrics_acc = tf.keras.metrics.CategoricalAccuracy()
    metrics_mcc = MCC(num_classes=5, name="mcc")
    # load weight and evalute
    model.load_weights("./modelSave/MDC_omics/BaseGCN_Attn_h7_fold%d\\ckpt"%num_fold)
    model.compile(optimizer=optimizer, loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])
   
    #get comfusion matrix
    y_pred_test, _ = model.predict(test_X)
    y_pred_test_argmax = tf.argmax(y_pred_test, axis=-1).numpy()
    cm = confusion_matrix(test_y.numpy(), y_pred_test_argmax)
    cm_label = ["LumA","LumB","Basal","Her2","Normal"]
    cm_df = pd.DataFrame(cm, index=cm_label, columns=cm_label)
    cm_dfs.append(cm_df)
    
    #attn extract
    model.evaluate(test_X, test_y_onehot)
    op_score_test = model.attn.get_attn_score(tf.transpose(test_X, [0,2,1]))
    model.evaluate(train_X, train_y_onehot)
    op_score_train = model.attn.get_attn_score(tf.transpose(train_X, [0,2,1]))
    
    head_score_all = []
    for i in range(7):
        single_head = tf.concat([op_score_train[i],op_score_test[i]], axis=0)
        head_score_all.append(single_head.numpy())
    
    # head_score_sum = []
    # for i in range(7):
    #     head_score_sum.append(head_score_all[i].sum(axis=0))
    
    # head_score_norm = []
    # for i in range(7):
    #     head_score_norm.append(head_score_all[i].sum(axis=0)/671)
    
    all_head_sum = np.zeros((671, 3,3))
    for i in range(7):    
        all_head_sum += head_score_all[i]
    all_head_sum /= 7
    
    omics_attn_score = all_head_sum.sum(axis=1)
    omics_attn_score_mean = omics_attn_score.mean(axis=0)
    omics_attn_score_std = omics_attn_score.std(axis=0)
    
    # get x_emd
    all_data = tf.concat([train_X, test_X], axis=0)
    all_y = tf.concat([train_y, test_y], axis=0).numpy()
    # all_y_onehot = tf.concat([train_y_onehot, test_y_onehot], axis=0).numpy()
    # model.evaluate(all_data, all_y_onehot)

    y_pred, acts = model.predict(all_data)
    x_emd_true = acts[0][np.argmax(y_pred,axis=-1)==all_y, :]
    x_emd_df = pd.DataFrame(x_emd_true, columns=nodes_id, index=patientID_cv.loc[np.argmax(y_pred,axis=-1)==all_y, "fold_%d"%num_fold])
    x_emd_df.to_csv("../Result/MDC_onehop_addRef_BaseGCN_attn_fold%d_emd.csv" % num_fold)
    
    # lrp
    y_pred, acts = model.predict(all_data)
    weights = [layer.weights[0].numpy() for layer in model.layers if "dense" in layer.name]
    lrp = LayerRelevancePropagation(weights=weights, activations=acts)
    lrp_score = lrp.relevance_propagation()
    lrp_score = lrp_score.numpy()
    
    lrp_score_df = pd.DataFrame(lrp_score, columns=nodes_id, index=patientID_cv["fold_%d"%num_fold])
    lrp_score_df["label"] = all_y
    lrp_score_df.to_csv("../Result/MDC_onehop_addRef_BaseGCN_attn_fold%d_lrp_score.csv" % num_fold)
   
    
    
#%% GCN+col_Attn+row_attn
cm_dfs = []
for num_fold in range(1,6):
    num_fold = 5
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
    
    model = BaseGCNModel_add_CRattn(gcn_hid_dim=32, 
                                    attn_embed_dim_1=256, attn_input_dim_1=train_data.shape[1], num_heads_1=7, 
                                    attn_embed_dim_2=64, attn_input_dim_2=32, num_heads_2=4,  
                                    fc_dim=512, output_dim=5,
                                    init_graph=graph_norm_dense,
                                    pool_method="max",
                                    dropout_rate=0.5)
    
    initial_learning_rate = 5*1e-3
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                  decay_steps=30,
                                                                  decay_rate=0.96,
                                                                  staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss_tracker = tf.keras.losses.CategoricalCrossentropy()
    metrics_auc = tf.keras.metrics.AUC(from_logits=False)
    metrics_acc = tf.keras.metrics.CategoricalAccuracy()
    metrics_mcc = MCC(num_classes=5, name="mcc")
    # load weight and evalute
    model.load_weights("./modelSave/BaseGCNModel_add_CRattn_fold%d\\ckpt"%num_fold)
    model.compile(optimizer=optimizer, loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])
    # model.build(input_shape=(None, train_X.shape[1], train_X.shape[2]))
    #get comfusion matrix
    y_pred_test, col_attn, row_attn = model.predict(test_X)
    y_pred_test_argmax = tf.argmax(y_pred_test[0], axis=-1).numpy()
    cm = confusion_matrix(test_y.numpy(), y_pred_test_argmax)
    cm_label = ["LumA","LumB","Basal","Her2","Normal"]
    cm_df = pd.DataFrame(cm, index=cm_label, columns=cm_label)
    # cm_dfs.append(cm_df)
    
    col_attn_norm = np.mean(col_attn, axis=0)    
    row_attn_norm = np.mean(row_attn, axis=0)
    row_attn_norm_df = pd.DataFrame(row_attn_norm)
    row_attn_norm_df.to_csv("../fold5_row_attn_norm.csv")
    
#%% valid datatset predict

data_file_path = "../Data/brca_pam50_hvG3167_valid.hdf5"
with h5py.File(data_file_path,"r") as f:
    val_data = f["valid_dataset"]["valid_data"][:,:,:].astype(np.float32)
    patient_ids = f["valid_dataset"].attrs["patient_ids"]
    graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]

graph_norm = normalize_adj(graph_data.astype(np.float32), False)
graph_norm_dense = tf.sparse.to_dense(graph_norm)

model = BaseGCNModel_addAttn(gcn_hid_dim=32,
                                attn_embed_dim=256, attn_out_dim=val_data.shape[1], 
                                num_heads=7,
                                fc_dim_1=512, fc_dim_2=256, output_dim=5, 
                                init_graph=graph_norm_dense,
                                dropout_rate=0.5,
                                pool_method="max")

initial_learning_rate = 5*1e-3
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                              decay_steps=30,
                                                              decay_rate=0.96,
                                                              staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_tracker = tf.keras.losses.CategoricalCrossentropy()
metrics_auc = tf.keras.metrics.AUC(from_logits=False)
metrics_acc = tf.keras.metrics.CategoricalAccuracy()
metrics_mcc = MCC(num_classes=5, name="mcc")
# load weight and evalute
model.load_weights("./modelSave/MDC_omics/BaseGCN_Attn_h7_fold5\\ckpt")
model.compile(optimizer=optimizer, loss=loss_tracker, metrics=[metrics_auc, metrics_acc, metrics_mcc])

convert_label = {"LumA":0, "LumB":1, "Basal":2, "Her2":3, "Normal":4}
y_pred_test, _ = model.predict(val_data)
result_df = pd.DataFrame(np.argmax(y_pred_test, axis=-1), index=patient_ids, columns=["pred_label"])
result_df.loc[result_df["pred_label"]==0, "PAM50"] = "LumA"
result_df.loc[result_df["pred_label"]==1, "PAM50"] = "LumB"
result_df.loc[result_df["pred_label"]==2, "PAM50"] = "Basal"
result_df.loc[result_df["pred_label"]==3, "PAM50"] = "Her2"
result_df.loc[result_df["pred_label"]==4, "PAM50"] = "Normal"

result_df = pd.concat([result_df, pd.DataFrame(y_pred_test, index=patient_ids)], axis=1)
