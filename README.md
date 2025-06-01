# Install
```
conda create -n scb_mult python=3.9 
conda activate scb_mult 

git clone git@github.com:Cai-fx/scb_multiome_demo.git
cd scb_multiome_demo
pip install . 
```

pybedtools may need `conda install bioconda::bedtools`

# Tutorial
Dataset: [10x multiomics pbmc](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0) 

Download h5 file from [link](https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5)

## 1. Data preprocessing
[preprocess.ipynb](tutorials/pbmc_10k/preprocess.ipynb)
- write atac, rna expression & sequence to preprocess folder. 
- (optional) prepare for benchmark
    - intersect atac peaks with chip data 
    - compute baseline metrics (spearmanR & motif matching score)

## 2. Model training
## pretrain with scbasset using only atac data
This step is very similar to scbasset, only replacing the final layer's bias with a linear func of atac depth 
```
cd tutorials/pbmc_10k/trials/scb_pretrain_intercept
. run.sh
```
## fine tune 
1. Initialize cnn with pretrained params
2. Fix cnn and train the mlp params
3. Train all the params
```
cd tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled
. run.sh
```

## 3. Model performance & Benchmark
[performance.ipynb](tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled/performance.ipynb)
- evaluate prediction accuracy
- visualize latent dimensions
- compute gradSeq & gradTF for TF activity, average over atac peaks

[benchmark.ipynb](tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled/benchmark.ipynb)
- evaluate TF binding prediction across peaks with cell type specific chip data 
- metrics:
    - baseline (2 dim): spearmanR, fimo_pval
    - standard (4 dim): spearmanR, fimo_pval, gradSeq, gradTF
    - extended (35 dim): spearmanR, fimo_pval, gradTF, peakemb
    - refless (1 dim): gradTF*gradSeq 

