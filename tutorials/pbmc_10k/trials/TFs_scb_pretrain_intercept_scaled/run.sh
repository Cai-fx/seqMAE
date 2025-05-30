#!/bin/bash
BASE_DIR="/mnt/fxcai/nfs_share2"

DATA_DIR="../../data"
atac_path="$DATA_DIR/atac_data/ad_atac_132k.h5ad"
preprocess_folder_acc="$DATA_DIR/atac_data/scb_processed"
preprocess_folder_rna="$DATA_DIR/rna_data"
RNA_FILE="$DATA_DIR/rna_data/tfs.h5ad"

pretrained_path="../scb_pretrain_intercept/result.pkl"

EXEC_PATH="../../../../run_scripts/run_scb_TFs.py"
python $EXEC_PATH --config config.json --rnapath $RNA_FILE --atacpath $atac_path --rna_pp_path $preprocess_folder_rna --atac_pp_path $preprocess_folder_acc --epochs 20 --pretrained_epochs 10 --pretrained_path $pretrained_path --scale_rna true