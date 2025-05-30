BASE_DIR="/mnt/fxcai/nfs_share2"

DATA_DIR="../../data"
atac_path="$DATA_DIR/atac_data/ad_atac_132k.h5ad"
preprocess_folder_acc="$DATA_DIR/atac_data/scb_processed"


EXEC_PATH="../../../../run_scripts/run_scb_pretrain.py"
python $EXEC_PATH --config config.json --epochs 500 --atacpath $atac_path --atac_pp_path $preprocess_folder_acc