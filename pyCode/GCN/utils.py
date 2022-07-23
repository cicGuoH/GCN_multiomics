# -*- coding: utf-8 -*-
"""
Created on Tue May 10 07:57:10 2022

@author: GUI
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf

def adj_to_sparse(A, is_sp):
    if is_sp:
        A_sp = sparse.csr_matrix(A)
    else:
        A_sp = A
    A_coo = A_sp.tocoo()
    inds = np.mat([A_coo.row, A_coo.col]).transpose()
    return tf.SparseTensor(inds, A_coo.data, A_coo.shape)


def normalize_adj(adj, is_sp):
    if is_sp:
        adj = adj
    else:
        adj = sparse.csr_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    adj_normalized_sp = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    adj_normalized_sp = adj_to_sparse(adj_normalized_sp, "sp_mat")
    return adj_normalized_sp
