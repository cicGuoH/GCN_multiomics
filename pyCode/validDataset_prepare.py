# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:40:27 2022

@author: GUI
"""


import pandas as pd
import numpy as np
import h5py
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import mygene
from sklearn.impute import SimpleImputer

# 准备预测的数据集

data_path = "../Data/BRCA_all_patients"

brca_com_data_m = pd.read_csv(data_path+"/BRCA_mRNA.csv", header=0, index_col=0)
brca_com_data_cnv = pd.read_csv(data_path+"/BRCA_CNV.csv", header=0, index_col=0)
brca_com_data_methy = pd.read_csv(data_path+"/BRCA_Methy.csv", header=0, index_col=0)
brca_sig_data_m = pd.read_csv("../Data/BRCA_significant/BRCA_mRNA.csv", header=0, index_col=0)

brca_pat_nolabel = list(set(brca_com_data_cnv.columns.values) - set(brca_sig_data_m.columns.values))

data_file_path = "../Data/brca_pam50_hvG3167_5cv.hdf5"
with h5py.File(data_file_path,"r") as f:
    nodes_id = f["PPI_network"]["Nodes"][:].astype("U")
    graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]    
    
brca_data_noLabel_m = brca_com_data_m[brca_pat_nolabel] 
brca_data_noLabel_cnv = brca_com_data_cnv[brca_pat_nolabel] 
brca_data_noLabel_methy = brca_com_data_methy[brca_pat_nolabel] 

brca_m_ppi = brca_data_noLabel_m.reindex(nodes_id, fill_value=np.nan)
brca_cnv_ppi = brca_data_noLabel_cnv.reindex(nodes_id, fill_value=np.nan)
brca_methy_ppi = brca_data_noLabel_methy.reindex(nodes_id, fill_value=np.nan)

brca_m_ppi = brca_m_ppi.T
brca_cnv_ppi = brca_cnv_ppi.T
brca_methy_ppi = brca_methy_ppi.T

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_m_ppi.min().min())
brca_m_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_m_ppi), columns=brca_m_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_cnv_ppi.min().min())
brca_cnv_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_cnv_ppi), columns=brca_cnv_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_methy_ppi.min().min())
brca_methy_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_methy_ppi), columns=brca_methy_ppi.columns, index=brca_m_ppi.index)

with h5py.File("../Data/brca_pam50_hvG3167_valid.hdf5", "w") as f:
    data_g = f.create_group("valid_dataset")
    data_g.attrs["patient_ids"] = brca_pat_nolabel
    data_ls = []
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("PPI_network_adjacency", data=graph_data, shape=graph_data.shape)
    
    for pat_id in brca_pat_nolabel:
        
        exp_dat = brca_m_ppi_fillna.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_fillna.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_fillna.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        data_ls.append(pat_feamap)
    data_ls = np.array(data_ls)
    data_g.create_dataset("valid_data", data=data_ls, shape=data_ls.shape)
