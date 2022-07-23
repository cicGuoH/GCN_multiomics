# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:49:10 2022

@author: GUI
"""

import pandas as pd
import numpy as np
from scipy import stats

convert_label = ["LumA","LumB","Basal","Her2","Normal"]
lrp_score_df = pd.read_csv("../Result/MDC_onehop_addRef_BaseGCN_attn_fold5_lrp_score.csv",
                           index_col=0, header=0)
patientID_cv = pd.read_csv("../patientID_CV.csv")


gene_ids = lrp_score_df.columns.values[:-1]
#分组得分统计
score_groups = pd.DataFrame(index=gene_ids)
for i in range(5):
    score_groups[convert_label[i]+"_mean"] = abs(lrp_score_df).loc[lrp_score_df["label"]==i, gene_ids].mean(axis=0)
    score_groups[convert_label[i]+"_std"] = abs(lrp_score_df).loc[lrp_score_df["label"]==i, gene_ids].std(axis=0)

# 分析每个患者的top100基因
top_genes_dfs = pd.DataFrame(index=lrp_score_df.index.values, columns=["top%d"%(i+1) for i in range(100)])
for pat in lrp_score_df.index.values:
    genes_rank = gene_ids[np.argsort(abs(lrp_score_df).loc[pat, gene_ids])[::-1]]
    top_genes_dfs.loc[pat,:] = genes_rank[:100]
    
top_genes = np.unique(top_genes_dfs.values.reshape(-1,))
result_df = pd.DataFrame(index=top_genes, columns=["count"])
for top_gene in top_genes:
    result_df.loc[top_gene, "count"] = sum(top_genes_dfs.values.reshape(-1,)==top_gene)


# val basal patients
val_lrp_score_df = lrp_score_df.iloc[-134:,:]
basal_val_lrp_score_df = val_lrp_score_df.loc[val_lrp_score_df["label"]==2,:]

top_genes_dfs = pd.DataFrame(index=basal_val_lrp_score_df.index.values, columns=["top%d"%(i+1) for i in range(100)])
for pat in basal_val_lrp_score_df.index.values:
    genes_rank = gene_ids[np.argsort(abs(basal_val_lrp_score_df).loc[pat, gene_ids])[::-1]]
    top_genes_dfs.loc[pat,:] = genes_rank[:100]

top_genes = np.unique(top_genes_dfs.values.reshape(-1,))
result_df = pd.DataFrame(index=top_genes, columns=["count"])
for top_gene in top_genes:
    result_df.loc[top_gene, "count"] = sum(top_genes_dfs.values.reshape(-1,)==top_gene)

  