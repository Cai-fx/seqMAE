# Install
```
conda create -n scb_mult python=3.9 
conda activate scb_mult 
conda install bioconda::bedtools 
pip install git+https://github.com/Cai-fx/scb_multiome_demo.git
```

# Tutorial
Dataset: [10x multiomics pbmc](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0) 

Download h5 file [here](https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5)

## 1. Data preprocessing
[preprocess.ipynb](tutorials/pbmc_10k/preprocess.ipynb)
- prepare atac, rna data for training
- (optional) prepare for benchmark
    - intersect atac peaks with chip data 
    - compute baseline metrics (spearmanR & motif matching score)

## 2. Model training
To train the model, modify run.sh files to include your data paths. 

## pretrain with scbasset using only atac data
```
cd tutorials/pbmc_10k/trials/scb_pretrain_intercept
. run.sh
```
## fine tune 
```
cd tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled
. run.sh
```

## 3. Model performance & Benchmark
[performance.ipynb](tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled/performance.ipynb)
- evaluate prediction accuracy
- visualize latent dimensions
- compute gradSeq & gradTF for TF activity (averaged over atac peaks)

[benchmark.ipynb](tutorials/pbmc_10k/trials/TFs_scb_pretrain_intercept_scaled/benchmark.ipynb)
- evaluate TF binding prediction with baseline, standard & extended (35 dim) model metrics with cell type-specific ChIP-seq data as ground truth 

