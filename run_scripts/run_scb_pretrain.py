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
    parser.add_argument('--outpath',
                        help="output .pkl file path",
                        type=str,
                        default="./result.pkl"
                        )
    parser.add_argument('--atacpath',
                        help="atac anndata file",
                        type=str)
    parser.add_argument('--atac_pp_path',
                        help="atac preprocess path",
                        type=str)

    parser.add_argument('--epochs', 
                        help="number of epochs",
                        type=int,
                        default=500)
    return parser


def main():
    
    parser = make_parser()
    arguments = parser.parse_args()
    config_d = json.load(arguments.config)
    
    ad_atac = sc.read_h5ad(arguments.atacpath)
    
    model = scbm.core.scb_pretrain.Model(preprocess_folder_acc = arguments.atac_pp_path,
                                    atac = ad_atac.X.T,
                                    model_config=config_d['model_config']
                                    )
    train_ds = model.read_accrna_ds(ds_key='train')
    val_ds = model.read_accrna_ds(ds_key='val')
    
    model.create_model()
    model.create_train_state(np.zeros((1, 1344, 4)),
                             train_ds['atac_dpth'],
                             learning_rate=1e-2,
                             pretrained_path=None, 
                             load_keys=["params_all"],
                             exclude_list=[]
                             )
    
    static_args = flax.core.freeze(config_d['static_args'])
    model.fit(n_epochs=arguments.epochs,
              num_train_batch=256,
              train_ds = train_ds,
              val_ds = val_ds,
              static_args=static_args,
              num_val_batch=64)
    
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