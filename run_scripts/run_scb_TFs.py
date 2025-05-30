import os
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import scb_multiome as scbm 
import numpy as np
import scanpy as sc 
import argparse
import json
import flax 



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        help="JSON file to be processed",
                        type=argparse.FileType('r'))
    parser.add_argument('--rnapath',
                        help="rna anndata file",
                        type=str)
    parser.add_argument('--atacpath',
                        help="atac anndata file",
                        type=str)
    parser.add_argument('--atac_pp_path',
                        help="atac preprocess path",
                        type=str)
    parser.add_argument('--rna_pp_path',
                        help="rna preprocess path",
                        type=str)
    
    parser.add_argument('--outpath',
                        help="output .pkl file path",
                        type=str,
                        default="./result.pkl"
                        )
    
    parser.add_argument('--epochs', 
                        help="number of epochs",
                        type=int,
                        default=500)
    parser.add_argument('--pretrained_path', 
                        help="params to init & fix in pretrain phase",
                        type=str,
                        default=None)
    parser.add_argument('--pretrained_epochs',
                        help="number of epochs for fixed CNN",
                        type=int,
                        default=0)
    parser.add_argument('--lr',
                        help="learning rate",
                        type=float,
                        default=0.01)
    
    parser.add_argument('--scale_rna',
                        help="sc.pp.scale",
                        type=bool,
                        default=False)
    
    return parser


def main():
    
    parser = make_parser()
    arguments = parser.parse_args()
    config_d = json.load(arguments.config)
    
    ad_atac = sc.read_h5ad(arguments.atacpath)
    ad_rna = sc.read_h5ad(arguments.rnapath)           #(normalized & log1p)      
    print(f"rna input shape: {ad_rna.shape}")  
    
    if arguments.scale_rna:
        print("scaling rna expression")
        sc.pp.scale(ad_rna)
    
    model = scbm.core.scb_TFs.Model(preprocess_folder_acc = arguments.atac_pp_path,
                                    preprocess_folder_rna = arguments.rna_pp_path,
                                    atac = ad_atac.X.T,
                                    rna = ad_rna.X,
                                    model_config=config_d['model_config']
                                    )
    train_ds = model.read_accrna_ds(ds_key='train')
    val_ds = model.read_accrna_ds(ds_key='val')
    
    model.create_model()
    model.create_train_state(np.zeros((1, 1344, 4)),
                             np.zeros_like(train_ds['rna']),
                             train_ds['atac_dpth'],
                             learning_rate=arguments.lr,
                             pretrained_path = arguments.pretrained_path,
                            #  pretrained_path=f"{BASE_DIR}/scb_multiome/pbmc_10k/trials/scb_pretrain_intercept/result.pkl",
                             load_keys=["params_all"],
                             exclude_list=["Z"]
                             )
    
    ## train with fixed cnn
    static_args = flax.core.freeze({**config_d['static_args'], "mlp_only":True})
    model.fit(n_epochs=arguments.pretrained_epochs,
              num_train_batch=256,
              train_ds = train_ds,
              val_ds = val_ds,
              static_args=static_args,
              num_val_batch=64,
              tidy_metrics=False)
    
    static_args = flax.core.freeze({**config_d['static_args'], "mlp_only":False})
    model.fit(n_epochs=arguments.epochs,
              num_train_batch=256,
              train_ds = train_ds,
              val_ds = val_ds,
              static_args=static_args,
              num_val_batch=64,
              tidy_metrics=True)
    
    all_ds = model.read_accrna_ds(ds_key='all')
    model.pred_y(ds=all_ds,
                 n_batch=256,
                 static_args=static_args)
    model.save_results(save_path=arguments.outpath, 
                       attr_keys=['params_all', 
                                    'y_pred', 
                                    'metrics'])
    
    return



if __name__=="__main__":
    main()