# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:39:39 2022

@author: GUI
"""

import pandas as pd
import numpy as np
import h5py
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import norm
import mygene
from sklearn.impute import SimpleImputer

# BRCA
brca_com_data_m = pd.read_csv("../Data/BRCA_complete/BRCA_mRNA.csv", header=0, index_col=0)
brca_com_data_cnv = pd.read_csv("../Data/BRCA_complete/BRCA_CNV.csv", header=0, index_col=0)
brca_com_data_mi = pd.read_csv("../Data/BRCA_complete/BRCA_miRNA.csv", header=0, index_col=0)
brca_com_data_methy = pd.read_csv("../Data/BRCA_complete/BRCA_Methy.csv", header=0, index_col=0)
brca_label = pd.read_csv("../Data/BRCA_complete/BRCA_label.csv", header=0, index_col=None)

# BRCA_sig
brca_sig_data_m = pd.read_csv("../Data/BRCA_significant/BRCA_mRNA.csv", header=0, index_col=0)
brca_sig_data_cnv = pd.read_csv("../Data/BRCA_significant/BRCA_CNV.csv", header=0, index_col=0)
brca_sig_data_mi = pd.read_csv("../Data/BRCA_significant/BRCA_miRNA.csv", header=0, index_col=0)
brca_sig_data_methy = pd.read_csv("../Data/BRCA_significant/BRCA_Methy.csv", header=0, index_col=0)


convert_label = {"LumA":0, "LumB":1, "Basal":2, "Her2":3, "Normal":4}
label_info = brca_label.copy()
label_info["Convert_label"] = np.nan
for k, v in convert_label.items():
    label_info.loc[label_info["Label"]==k, "Convert_label"] = v
    
#%% network process
ppi_raw_df = pd.read_csv("..\Data\9606.protein.links.v11.5.txt", sep=" ",header=0, index_col=None)
high_cf_ppi = ppi_raw_df[ppi_raw_df.combined_score >= 850]
high_cf_ppi["protein1"] = high_cf_ppi["protein1"].apply(lambda x: x.split(".")[1])
high_cf_ppi["protein2"] = high_cf_ppi["protein2"].apply(lambda x: x.split(".")[1])

plt.figure(figsize=(9,6), dpi=300)
sns.distplot(ppi_raw_df.combined_score,
             hist=True,
             bins=100,
             kde=True,
             kde_kws={"bw":0.1},
             # hist_kws={'histtype':"bar"}, #默认为bar,可选barstacked,step,stepfilled
             color="#098154")
plt.title("Distribution of Combined score among protein interactions")
plt.show()


def get_gene_symbols_from_proteins(list_of_ensembl_ids):
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_ensembl_ids,
                       scopes='ensembl.protein',
                       fields='symbol',
                       species='human', returnall=True
                      )

    def get_symbol_and_ensembl(d):
        if 'symbol' in d:
            return [d['query'], d['symbol']]
        else:
            return [d['query'], None]

    node_names = [get_symbol_and_ensembl(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=['Ensembl_ID', 'Symbol']).set_index('Ensembl_ID')
    node_names.dropna(axis=0, inplace=True)
    return node_names

ens_names_all = ppi_raw_df.protein1.append(ppi_raw_df.protein2).unique()
ens_names = high_cf_ppi.protein1.append(high_cf_ppi.protein2).unique()
ens_to_symbol = get_gene_symbols_from_proteins(ens_names)


ens_names_no_hit = list(set(ens_names).difference(set(ens_to_symbol.index)))
with open("../ens_no_hit.txt","w") as f:
    for i in ens_names_no_hit:
        f.write(i+"\n")
        


p1_incl = high_cf_ppi.join(ens_to_symbol, on='protein1', how='inner', rsuffix='_p1')
both_incl = p1_incl.join(ens_to_symbol, on='protein2', how='inner', rsuffix='_p2')
string_edgelist_symbols = both_incl.drop(['protein1', 'protein2'], axis=1)
string_edgelist_symbols.columns = ['confidence', 'partner1', 'partner2']
string_ppi_final = string_edgelist_symbols[['partner1', 'partner2', 'confidence']]

G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
string_ppi_final.to_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index=None)

# %% Genes determine
ppi_graph = nx.from_pandas_edgelist(df=string_ppi_final, source="partner1", target="partner2", edge_attr="confidence")
ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
ppi_network_edgelist = nx.to_pandas_edgelist(ppi_graph)

# compute shortest path length
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
length = dict(nx.all_pairs_shortest_path_length(G))
nodes_ls = list(G.nodes)
graph_raw_data = nx.to_pandas_adjacency(G=G)
# nx.is_connected(G)

tmp_arr = np.ones((len(nodes_ls), len(nodes_ls)))
for i, node_i in enumerate(nodes_ls):
    for j, node_j in enumerate(nodes_ls):
        try:
            tmp_arr[i,j] = length[node_i][node_j]
        except:
            tmp_arr[i,j] = np.inf
spl_df = pd.DataFrame(tmp_arr, index=nodes_ls, columns=nodes_ls)
spl_df.to_csv("../Data/shortest_path_length_highConf_ppi.csv")


spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)

inter_genes = set(brca_sig_data_m.index) | set(brca_sig_data_methy.index) | set(brca_sig_data_cnv.index) 
hvg_ppi_genes = list(set(inter_genes) & set(nodes_ls))

high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=5), 1, 0)
ro = (high_variance_genes_adj_arr.sum()-3167)/3167/(3166)
print(ro, (high_variance_genes_adj_arr.sum()-3167)/2)
#reindex according to ppi index
#high variable genes
brca_m_ppi = brca_com_data_m.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_cnv_ppi = brca_com_data_cnv.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_methy_ppi = brca_com_data_methy.reindex(hvg_ppi_genes, fill_value=np.nan)

methy_nodes = brca_methy_ppi.shape[0] -  pd.isna(brca_methy_ppi.iloc[:,0]).sum()
m_nodes = brca_m_ppi.shape[0] -  pd.isna(brca_m_ppi.iloc[:,0]).sum()
cnv_nodes = brca_cnv_ppi.shape[0] -  pd.isna(brca_cnv_ppi.iloc[:,0]).sum()
print ("Network has {} nodes/genes".format(len(hvg_ppi_genes)))
print ("* {}  genes in network have methylation data".format(methy_nodes))
print ("* {} genes in network have gene expression".format(m_nodes))
print ("* {} genes in network have CNA information".format(cnv_nodes))
# transpose 
brca_m_ppi = brca_m_ppi.T
brca_cnv_ppi = brca_cnv_ppi.T
brca_methy_ppi = brca_methy_ppi.T

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_m_ppi.min().min())
brca_m_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_m_ppi), columns=brca_m_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_cnv_ppi.min().min())
brca_cnv_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_cnv_ppi), columns=brca_cnv_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_methy_ppi.min().min())
brca_methy_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_methy_ppi), columns=brca_methy_ppi.columns, index=brca_m_ppi.index)


pat_ids = brca_m_ppi.index.values
data_index = [i for i in range(brca_m_ppi.shape[0])]
train_index, test_index, _, _ = train_test_split(data_index, label_info["Convert_label"], test_size=0.2, random_state=1234)
train_patient_ids = pat_ids[train_index]
test_patient_ids = pat_ids[test_index]    

brca_m_ppi_train = brca_m_ppi_fillna.loc[train_patient_ids,:]
brca_cnv_ppi_train = brca_cnv_ppi_fillna.loc[train_patient_ids,:]
brca_methy_ppi_train = brca_methy_ppi_fillna.loc[train_patient_ids,:]
brca_m_ppi_test = brca_m_ppi_fillna.loc[test_patient_ids,:]
brca_cnv_ppi_test = brca_cnv_ppi_fillna.loc[test_patient_ids,:]
brca_methy_ppi_test = brca_methy_ppi_fillna.loc[test_patient_ids,:]

#%% add reference genes
gda_info = pd.read_csv("../Data/C0678222_disease_gda_summary.tsv", sep="\t")
gda_info_sub = gda_info.loc[:,["Gene","Score_gda"]]
gda_info_sub = gda_info_sub.set_index("Gene")
gda_info_sub2 = gda_info_sub.loc[gda_info_sub["Score_gda"]>0.2,:]

inter_genes = set(brca_sig_data_m.index) | set(brca_sig_data_methy.index) | set(brca_sig_data_cnv.index) | set(gda_info_sub2.index)
hvg_ppi_genes = list(set(inter_genes) & set(nodes_ls))

high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=1), 1, 0)
ro = (high_variance_genes_adj_arr.sum()-len(hvg_ppi_genes))/len(hvg_ppi_genes)/(len(hvg_ppi_genes)-1)
print(ro, (high_variance_genes_adj_arr.sum()-len(hvg_ppi_genes))/2)
#reindex according to ppi index
#high variable genes
brca_m_ppi = brca_com_data_m.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_cnv_ppi = brca_com_data_cnv.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_methy_ppi = brca_com_data_methy.reindex(hvg_ppi_genes, fill_value=np.nan)

methy_nodes = brca_methy_ppi.shape[0] -  pd.isna(brca_methy_ppi.iloc[:,0]).sum()
m_nodes = brca_m_ppi.shape[0] -  pd.isna(brca_m_ppi.iloc[:,0]).sum()
cnv_nodes = brca_cnv_ppi.shape[0] -  pd.isna(brca_cnv_ppi.iloc[:,0]).sum()
print ("Network has {} nodes/genes".format(len(hvg_ppi_genes)))
print ("* {}  genes in network have methylation data".format(methy_nodes))
print ("* {} genes in network have gene expression".format(m_nodes))
print ("* {} genes in network have CNA information".format(cnv_nodes))

G = nx.from_pandas_adjacency(high_variance_genes_adj)
G = nx.write_edgelist(G, path="../Data/brca_pam50_hvG3483_onehop.edgelist", data=False)

# transpose 
brca_m_ppi = brca_m_ppi.T
brca_cnv_ppi = brca_cnv_ppi.T
brca_methy_ppi = brca_methy_ppi.T

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_m_ppi.min().min())
brca_m_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_m_ppi), columns=brca_m_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_cnv_ppi.min().min())
brca_cnv_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_cnv_ppi), columns=brca_cnv_ppi.columns, index=brca_m_ppi.index)

imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_methy_ppi.min().min())
brca_methy_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_methy_ppi), columns=brca_methy_ppi.columns, index=brca_m_ppi.index)

cv_dataset = {}
pat_ids = brca_m_ppi.index.values
data_index = [i for i in range(brca_sig_data_m.shape[1])]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_m_ppi_train = brca_m_ppi_fillna.loc[train_patient_ids,:]
    brca_cnv_ppi_train = brca_cnv_ppi_fillna.loc[train_patient_ids,:]
    brca_methy_ppi_train = brca_methy_ppi_fillna.loc[train_patient_ids,:]
    brca_m_ppi_test = brca_m_ppi_fillna.loc[test_patient_ids,:]
    brca_cnv_ppi_test = brca_cnv_ppi_fillna.loc[test_patient_ids,:]
    brca_methy_ppi_test = brca_methy_ppi_fillna.loc[test_patient_ids,:]
    
    tr_ls = []
    for pat_id in train_patient_ids:
        exp_dat = brca_m_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    
    te_ls = []
    for pat_id in test_patient_ids:
        exp_dat = brca_m_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (tr_ls, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (te_ls, label_info.loc[test_ind,"Convert_label"].values)


with h5py.File("../Data/brca_pam50_hvG3483_3hop_5cv.hdf5", "w") as f:
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
    
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        

#%% hvG3167 train-test dataset
with h5py.File("../Data/brca_pam50_hvG3167.hdf5", "w") as f:
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=np.array(hvg_ppi_genes), shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
    
    tr_g = f.create_group("Tr_dataset")
    tr_g.attrs["patient_ids"] = train_index
    tr_ls = []
    
    for pat_id in train_patient_ids:
        
        exp_dat = brca_m_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    tr_g.create_dataset("tr_data", data=tr_ls, shape=tr_ls.shape)
    
    label_to_save = tr_g.create_group("tr_pam50")
    label_to_save.create_dataset("PAM50", 
                              data=label_info.loc[train_index,"Convert_label"].values, 
                              shape=label_info.loc[train_index,"Convert_label"].values.shape)
    
    te_g = f.create_group("Te_dataset")
    te_g.attrs["patient_ids"] = test_patient_ids
    te_ls = []
    for pat_id in test_patient_ids:
        exp_dat = brca_m_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    te_g.create_dataset("te_data", data=te_ls, shape=te_ls.shape)
        
    label_to_save = te_g.create_group("te_pam50")
    label_to_save.create_dataset("PAM50", 
                              data=label_info.loc[test_index,"Convert_label"].values, 
                              shape=label_info.loc[test_index,"Convert_label"].values.shape)
    
#%% brca_pam50_hvG3167_5cv
# 5 fold dataset prepare
cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_m_ppi_train = brca_m_ppi_fillna.loc[train_patient_ids,:]
    brca_cnv_ppi_train = brca_cnv_ppi_fillna.loc[train_patient_ids,:]
    brca_methy_ppi_train = brca_methy_ppi_fillna.loc[train_patient_ids,:]
    brca_m_ppi_test = brca_m_ppi_fillna.loc[test_patient_ids,:]
    brca_cnv_ppi_test = brca_cnv_ppi_fillna.loc[test_patient_ids,:]
    brca_methy_ppi_test = brca_methy_ppi_fillna.loc[test_patient_ids,:]
    
    tr_ls = []
    for pat_id in train_patient_ids:
        exp_dat = brca_m_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    
    te_ls = []
    for pat_id in test_patient_ids:
        exp_dat = brca_m_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat, cnv_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (tr_ls, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (te_ls, label_info.loc[test_ind,"Convert_label"].values)


with h5py.File("../Data/brca_pam50_hvG3167_5hop_5cv.hdf5", "w") as f:
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
    
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)

#%% for ML training, concat 
# 5 fold dataset prepare
cv_dataset = {}
pat_ids = brca_sig_data_m.columns.values
data_index = [i for i in range(brca_sig_data_m.shape[1])]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_sig_data_m_train = brca_sig_data_m.T.loc[train_patient_ids,:]
    brca_sig_data_cnv_train = brca_sig_data_cnv.T.loc[train_patient_ids,:]
    brca_sig_data_methy_train = brca_sig_data_methy.T.loc[train_patient_ids,:]
    brca_sig_data_m_test = brca_sig_data_m.T.loc[test_patient_ids,:]
    brca_sig_data_cnv_test = brca_sig_data_cnv.T.loc[test_patient_ids,:]
    brca_sig_data_methy_test = brca_sig_data_methy.T.loc[test_patient_ids,:]
    
    tr_dat = pd.concat([brca_sig_data_m_train,brca_sig_data_cnv_train,brca_sig_data_methy_train], axis=1).values
    te_dat = pd.concat([brca_sig_data_m_test,brca_sig_data_cnv_test,brca_sig_data_methy_test], axis=1).values
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_dat, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_dat, label_info.loc[test_ind,"Convert_label"].values)
    

with h5py.File("../Data/brca_pam50_concat5974_5cv.hdf5", "w") as f:
    # save network
    
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)
        
        
# %% mRNA_hvG1237 Dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
hvg_ppi_genes = list(set(brca_sig_data_m.index.tolist()) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)
brca_m_ppi = brca_com_data_m.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_m_ppi = brca_m_ppi.T
pat_ids = brca_m_ppi.index.values
data_index = [i for i in range(brca_m_ppi.shape[0])]

cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_sig_data_m_train = brca_m_ppi.loc[train_patient_ids,:]
    brca_sig_data_m_test = brca_m_ppi.loc[test_patient_ids,:]
    
    tr_dat = brca_sig_data_m_train.values
    te_dat = brca_sig_data_m_test.values
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_dat, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_dat, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_MhvG1237_5cv.hdf5", "w") as f:
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)
        
#%% DNAmethy_hvG dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
hvg_ppi_genes = list(set(brca_sig_data_methy.index.tolist()) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)
brca_methy_ppi = brca_sig_data_methy.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_methy_ppi = brca_methy_ppi.T
pat_ids = brca_methy_ppi.index.values
data_index = [i for i in range(brca_methy_ppi.shape[0])]

cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_sig_data_methy_train = brca_methy_ppi.loc[train_patient_ids,:]
    brca_sig_data_methy_test = brca_methy_ppi.loc[test_patient_ids,:]
    
    tr_dat = brca_sig_data_methy_train.values
    te_dat = brca_sig_data_methy_test.values
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_dat, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_dat, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_MethyhvG1070_5cv.hdf5", "w") as f:
    # save network
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)
        
#%% CNV_hvG dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
hvg_ppi_genes = list(set(brca_sig_data_cnv.index.tolist()) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)
brca_cnv_ppi = brca_sig_data_cnv.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_cnv_ppi = brca_cnv_ppi.T
pat_ids = brca_cnv_ppi.index.values
data_index = [i for i in range(brca_cnv_ppi.shape[0])]

cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    brca_sig_data_cnv_train = brca_cnv_ppi.loc[train_patient_ids,:]
    brca_sig_data_cnv_test = brca_cnv_ppi.loc[test_patient_ids,:]
    
    tr_dat = brca_sig_data_cnv_train.values
    te_dat = brca_sig_data_cnv_test.values
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_dat, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_dat, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_cnvhvG1198_5cv.hdf5", "w") as f:
    # save network
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)

#%% mRNA_DNAmethy_hvG dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
inter_genes = set(brca_sig_data_m.index) | set(brca_sig_data_methy.index) 
hvg_ppi_genes = list(set(inter_genes) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)

brca_m_ppi = brca_com_data_m.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_methy_ppi = brca_com_data_methy.reindex(hvg_ppi_genes, fill_value=np.nan)

brca_m_ppi = brca_m_ppi.T
brca_methy_ppi = brca_methy_ppi.T

pat_ids = brca_cnv_ppi.index.values
data_index = [i for i in range(brca_cnv_ppi.shape[0])]


imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_m_ppi.min().min())
brca_m_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_m_ppi), columns=brca_m_ppi.columns, index=brca_m_ppi.index)
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_methy_ppi.min().min())
brca_methy_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_methy_ppi), columns=brca_methy_ppi.columns, index=brca_m_ppi.index)


cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    
    brca_m_ppi_train = brca_m_ppi_fillna.loc[train_patient_ids,:]
    brca_methy_ppi_train = brca_methy_ppi_fillna.loc[train_patient_ids,:]
    brca_m_ppi_test = brca_m_ppi_fillna.loc[test_patient_ids,:]
    brca_methy_ppi_test = brca_methy_ppi_fillna.loc[test_patient_ids,:]
    
    tr_ls = []
    for pat_id in train_patient_ids:
        exp_dat = brca_m_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    
    te_ls = []
    for pat_id in test_patient_ids:
        exp_dat = brca_m_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        methy_dat = brca_methy_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, methy_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_ls, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_ls, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_mRNA_DNAMethy_hvG2200_5cv.hdf5", "w") as f:
    # save network
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)

#%% mRNA_CNV_hvG dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
inter_genes = set(brca_sig_data_m.index) | set(brca_sig_data_cnv.index) 
hvg_ppi_genes = list(set(inter_genes) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)

brca_m_ppi = brca_com_data_m.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_cnv_ppi = brca_com_data_cnv.reindex(hvg_ppi_genes, fill_value=np.nan)

brca_m_ppi = brca_m_ppi.T
brca_cnv_ppi = brca_cnv_ppi.T

pat_ids = brca_cnv_ppi.index.values
data_index = [i for i in range(brca_cnv_ppi.shape[0])]


imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_m_ppi.min().min())
brca_m_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_m_ppi), columns=brca_m_ppi.columns, index=brca_m_ppi.index)
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_cnv_ppi.min().min())
brca_cnv_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_cnv_ppi), columns=brca_cnv_ppi.columns, index=brca_m_ppi.index)


cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    
    brca_m_ppi_train = brca_m_ppi_fillna.loc[train_patient_ids,:]
    brca_cnv_ppi_train = brca_cnv_ppi_fillna.loc[train_patient_ids,:]
    brca_m_ppi_test = brca_m_ppi_fillna.loc[test_patient_ids,:]
    brca_cnv_ppi_test = brca_cnv_ppi_fillna.loc[test_patient_ids,:]
    
    tr_ls = []
    for pat_id in train_patient_ids:
        exp_dat = brca_m_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, cnv_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    
    te_ls = []
    for pat_id in test_patient_ids:
        exp_dat = brca_m_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([exp_dat, cnv_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_ls, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_ls, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_mRNA_cnv_hvG2294_5cv.hdf5", "w") as f:
    # save network
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)

#%% DNAMethy_CNV_hvG dataset
spl_df = pd.read_csv("../Data/shortest_path_length_highConf_ppi.csv", header=0, index_col=0)
string_ppi_final = pd.read_csv('../Data/High_Confidence850_StringDB.tsv', sep='\t', index_col=None)
G = nx.from_pandas_edgelist(string_ppi_final, source='partner1', target='partner2')
nodes_ls = list(G.nodes)
inter_genes = set(brca_sig_data_methy.index) | set(brca_sig_data_cnv.index) 
hvg_ppi_genes = list(set(inter_genes) & set(nodes_ls))
high_variance_genes_adj = spl_df.loc[hvg_ppi_genes, hvg_ppi_genes]      
high_variance_genes_adj_arr = np.where((high_variance_genes_adj.values<=2), 1, 0)

brca_methy_ppi = brca_com_data_methy.reindex(hvg_ppi_genes, fill_value=np.nan)
brca_cnv_ppi = brca_com_data_cnv.reindex(hvg_ppi_genes, fill_value=np.nan)

brca_methy_ppi = brca_methy_ppi.T
brca_cnv_ppi = brca_cnv_ppi.T

pat_ids = brca_cnv_ppi.index.values
data_index = [i for i in range(brca_cnv_ppi.shape[0])]


imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_methy_ppi.min().min())
brca_methy_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_methy_ppi), columns=brca_methy_ppi.columns, index=brca_methy_ppi.index)
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=brca_cnv_ppi.min().min())
brca_cnv_ppi_fillna = pd.DataFrame(imp.fit_transform(brca_cnv_ppi), columns=brca_cnv_ppi.columns, index=brca_cnv_ppi.index)


cv_dataset = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123123123)
for i, (train_ind, test_ind) in enumerate(skf.split(data_index, label_info["Convert_label"])):
    train_patient_ids = pat_ids[train_ind]
    test_patient_ids = pat_ids[test_ind] 
    
    brca_methy_ppi_train = brca_methy_ppi_fillna.loc[train_patient_ids,:]
    brca_cnv_ppi_train = brca_cnv_ppi_fillna.loc[train_patient_ids,:]
    brca_methy_ppi_test = brca_methy_ppi_fillna.loc[test_patient_ids,:]
    brca_cnv_ppi_test = brca_cnv_ppi_fillna.loc[test_patient_ids,:]
    
    tr_ls = []
    for pat_id in train_patient_ids:
        methy_dat = brca_methy_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_train.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([methy_dat, cnv_dat], axis=1)
        tr_ls.append(pat_feamap)
    tr_ls = np.array(tr_ls)
    
    te_ls = []
    for pat_id in test_patient_ids:
        methy_dat = brca_methy_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        cnv_dat = brca_cnv_ppi_test.loc[pat_id, :].values.reshape((-1, 1))
        pat_feamap = np.concatenate([methy_dat, cnv_dat], axis=1)
        te_ls.append(pat_feamap)
    te_ls = np.array(te_ls)
    
    cv_dataset["fold_%d" % (i+1)] = {}
    cv_dataset["fold_%d" % (i+1)]["train"] = (train_patient_ids, tr_ls, label_info.loc[train_ind,"Convert_label"].values)
    cv_dataset["fold_%d" % (i+1)]["test"] = (test_patient_ids, te_ls, label_info.loc[test_ind,"Convert_label"].values)
    
with h5py.File("../Data/brca_pam50_methy_cnv_hvG2162_5cv.hdf5", "w") as f:
    # save network
    # save network
    net_g = f.create_group("PPI_network")
    net_g.create_dataset("Nodes", data=hvg_ppi_genes, shape=len(hvg_ppi_genes))
    net_g.create_dataset("PPI_network_adjacency", data=high_variance_genes_adj_arr, shape=high_variance_genes_adj_arr.shape)
   
    for i in range(1,6):
        data_single_group = f.create_group("fold_%d"%i)
        data_single_group.create_dataset("train_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["train"][0],
                                         shape=cv_dataset["fold_%d"%i]["train"][0].shape)
        data_single_group.create_dataset("train_data",
                                         data=cv_dataset["fold_%d"%i]["train"][1],
                                         shape=cv_dataset["fold_%d"%i]["train"][1].shape)
        data_single_group.create_dataset("train_label",
                                         data=cv_dataset["fold_%d"%i]["train"][2],
                                         shape=cv_dataset["fold_%d"%i]["train"][2].shape)
        data_single_group.create_dataset("test_pat_ids",
                                         data=cv_dataset["fold_%d"%i]["test"][0],
                                         shape=cv_dataset["fold_%d"%i]["test"][0].shape)
        data_single_group.create_dataset("test_data",
                                         data=cv_dataset["fold_%d"%i]["test"][1],
                                         shape=cv_dataset["fold_%d"%i]["test"][1].shape)
        data_single_group.create_dataset("test_label",
                                         data=cv_dataset["fold_%d"%i]["test"][2],
                                         shape=cv_dataset["fold_%d"%i]["test"][2].shape)
