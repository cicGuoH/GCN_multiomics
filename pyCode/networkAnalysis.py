# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:09:41 2022

@author: GUI
"""

import networkx as nx
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.sans-serif"] = "Arial"
import upsetplot

data_file_path = "../Data/brca_pam50_hvG3167_5cv.hdf5"
with h5py.File(data_file_path,"r") as f:
    nodes_id = f["PPI_network"]["Nodes"][:].astype("U")
    graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]

graph_data_df = pd.DataFrame(graph_data-np.eye(graph_data.shape[0]), index=nodes_id, columns=nodes_id)
G = nx.from_pandas_adjacency(graph_data_df)

nx.write_edgelist(G, path="../Data/brca_pam50_hvG3167.edgelist", data=False)

row_attn_norm_df = pd.read_csv("../fold5_row_attn_norm.csv",header=0, index_col=0)
row_attn_norm_df.index = nodes_id
row_attn_norm_df.columns = nodes_id

node_degree = graph_data.sum(axis=1)
node_score_sum = row_attn_norm_df.values.sum(axis=0)
node_score = row_attn_norm_df.values.sum(axis=0) / node_degree
node_score_max = row_attn_norm_df.values.max(axis=0) 



plt.figure(figsize=(6,4), dpi=300)
sns.distplot(node_degree,
             hist=True,
             bins=100,
             kde=True,
             kde_kws={"bw":0.1},
             # hist_kws={'histtype':"bar"}, #默认为bar,可选barstacked,step,stepfilled
             color="#098154")
plt.title("Distribution of Degree-hvGGI")
plt.show()


#%% 分析两种网络Degree Top N基因的GDA得分
subset_gene_info = pd.read_csv("../Result/NetworkAnalysis/brca_pam50_hvG3167_subset_node_info.csv")
hop2_gene_info = pd.read_csv("../Result/NetworkAnalysis/brca_pam50_hvG3167_node_info.csv")
gda_info = pd.read_csv("../Data/C0678222_disease_gda_summary.tsv", sep="\t")
subset_gene_info_sub = subset_gene_info.loc[:,["name","Degree"]]
subset_gene_info_sub = subset_gene_info_sub.set_index("name")
hop2_gene_info_sub = hop2_gene_info.loc[:,["name","Degree"]]
hop2_gene_info_sub = hop2_gene_info_sub.set_index("name")
gda_info_sub = gda_info.loc[:,["Gene","Score_gda"]]
gda_info_sub = gda_info_sub.set_index("Gene")
gda_info_sub2 = gda_info_sub.loc[gda_info_sub["Score_gda"]>0.2,:]
subset_genes = set(subset_gene_info_sub.index.tolist()) & set(gda_info_sub2.index.tolist())
subset_gene_info_sub.loc[subset_genes, "GDA"] = gda_info_sub.loc[subset_genes, ["Score_gda"]].values

hop2_genes = set(hop2_gene_info_sub.index.tolist()) & set(gda_info_sub2.index.tolist())
hop2_gene_info_sub.loc[hop2_genes, "GDA"] = gda_info_sub.loc[hop2_genes, ["Score_gda"]].values

data_file_path = "../Data/brca_pam50_hvG3167_5cv.hdf5"
with h5py.File(data_file_path,"r") as f:
    nodes_id = f["PPI_network"]["Nodes"][:].astype("U")
    # graph_data = f["PPI_network"]["PPI_network_adjacency"][:,:]

gda_nodes = set(gda_info_sub2.index)
gda_nodes_ppi = set(nodes_id) & gda_nodes

def get_top_ratio(ratio, hop2_gene_info_sub=hop2_gene_info_sub, subset_gene_info_sub=subset_gene_info_sub):
    hop2_gene_info_sub = hop2_gene_info_sub.sort_values(by="Degree")
    subset_gene_info_sub = subset_gene_info_sub.sort_values(by="Degree")
    hop2_top_genes = hop2_gene_info_sub.index.values[:ratio]
    subset_top_genes = subset_gene_info_sub.index.values[:ratio]
    
    hop2_num = len(set(hop2_top_genes) & gda_nodes_ppi)
    subset_num = len(set(subset_top_genes) & gda_nodes_ppi)
    print("hop2:%d, subset:%d" % (hop2_num, subset_num))

get_top_ratio(400)

#%% upsetplot 可视化4组基因的交集
data = pd.read_excel("../文章/tables/高变基因与参考基因的交集.xlsx", header=0, index_col=None)
content = {
    "mRNA":list(data["mRNA"].dropna()),
    "DNA methylation":list(data["DNAMethy"].dropna()),
    "CNV":list(data["CNV"].dropna()),
    "ref.Genes":list(data["ref. Genes"].dropna())
    }
fig = plt.figure(figsize=(8.,5),dpi=300)
upsetplot.plot(upsetplot.from_contents(content), fig=fig, element_size=25, facecolor="#e29578", subset_size="count")
plt.savefig("../文章/figures/upsetplot.png", bbox_inches="tight")
plt.show()