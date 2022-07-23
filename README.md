# Attention-based graph neural network (AGCN) for multi-omics integration

![overview](https://user-images.githubusercontent.com/108866548/180596142-78b0b8c7-756a-4f92-bc77-7b3d919c077a.png)

Based on multi-omics data from TCGA BRCA, we proposed an attention-based GCN for classification of different breast cancer molecular subtypes and jointly learning associations among multiple molecular levels of gene. We compared the model performance of GCN with three types of attention mechanism approaches. The results show that GCN combined with omics-specific attention mechanism can significantly improve the prediction performance of neural network. We also analyzed the prediction performance under different molecular-level data types, and proved that attention-based GCN has certain advantages in integrating high-dimensional features. Furthermore, we adopted LRP algorithm to explain model decisions and identify patient-specific gene markers and related functional modules. 

Requirements:
- Python 3.7
- Tensorflow 2.8
- scikit-learn
- networkx 2.6.3



