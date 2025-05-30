from typing import Any, Tuple, Optional
from jax._src.prng import PRNGKeyArray 
import flax.linen as nn
import optax
import jax.numpy as jnp
from functools import partial
import h5py
import jax
import flax 

from ..utils.scb_blocks import MLP, Seq2PeakEmb_orig
from ..utils.train_utils import trainState 

from ..utils.IOs import seq_rna_generator

import numpy as np
import jax.numpy as jnp
from .base_model import BaseModel
from ..utils.eval_metrics import compute_metrics_rna, compute_metrics_acc
from tqdm import tqdm
from typing import Callable
from flax.core.frozen_dict import FrozenDict
import logging 



class scb_bulk_classifier(nn.Module):
    latent_dim : int = 32 
    infer_filter_num : int = 288 
    kernel_sizes : tuple = (17, 5, 5, 5, 5, 5, 5, 1)
    pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    activation_func : Callable = nn.gelu
    n_rng : int = 3
    
    @nn.compact
    def __call__(self, seq, train):
        seq = Seq2PeakEmb_orig(latent_dim = self.latent_dim,
                                known_filter = False,
                                infer_filter_num = self.infer_filter_num,
                                kernel_sizes = self.kernel_sizes,
                                pool_sizes = self.pool_sizes,
                                layer_features  = self.layer_features,
                                activation_func = self.activation_func
                                )(seq, 0, train)                        #(n_peak, d)
        
        seq = nn.Dense(features=1,
                       use_bias=True)(seq)
        return seq


    
class Model(BaseModel):
    def __init__(self, 
                 preprocess_folder_acc = None,
                 chip_bulk = None,
                 model_config = {}, 
                 **kwargs):
        # chip_bulk: (n_peak, )
        self.preprocess_folder_acc = preprocess_folder_acc
        self.chip_bulk = chip_bulk
        self.model_config = model_config
        super().__init__(**kwargs)
        return 

    def create_model(self):
        self.model = scb_bulk_classifier(**self.model_config)
        
        return
    
    def read_accrna_ds(self, ds_key='train', **kwargs) -> dict:   
        preprocess_folder = self.preprocess_folder_acc
        m = self.chip_bulk                                  ##(n_peak, )
        train_data = '%s/%s_seqs.h5'%(preprocess_folder, ds_key)
        split_file = '%s/splits.h5'%preprocess_folder
                
        if ds_key == 'all':
            train_ids = np.arange(m.shape[0])
        else:    
            with h5py.File(split_file, 'r') as hf:
                train_ids = hf[f'{ds_key}_ids'][:]

        m_train = m[train_ids]
        
        ds = seq_rna_generator(train_data, m_train)()
        ds = np.array([_x for _x in ds], dtype=object)
        train_x = np.array(list(zip(*ds))[0])
        train_y = np.array(list(zip(*ds))[1])

        return {'x_acc': train_x, 'y_chip': train_y} 
            
    
    def shuffle_batch(self, rng : Optional[PRNGKeyArray], ds : dict, num_batch : int) -> Tuple:
        ds_size_acc = ds['x_acc'].shape[0]              #n_peak
        
        perms_acc = self._shuffle_batch(rng, ds_size_acc, num_batch)        #(n_batch, batch_size)
        perms_rna = jnp.zeros((perms_acc.shape[0], 1))
        return perms_acc, perms_rna
    
    def create_batch(self, perm, ds: dict) -> dict:
        perm_acc, _ = perm
        batch = {'x_acc': ds['x_acc'][perm_acc,:],
                'y_chip': ds['y_chip'][perm_acc,:],
                }
        return batch
    
    @classmethod
    @partial(jax.jit, static_argnames=["self", "static_args"])
    def train_step_func(self, 
                        state : trainState, 
                        batch : dict, 
                        static_args : FrozenDict, 
                        *rngs) -> Tuple[trainState, dict]:
        '''
        performs gradient descend for each minibatch, jitted function
        
        Returns:
            new_state, metrics     
        '''
        grad_fn = jax.value_and_grad(loss, has_aux=True, argnums=0) 
        (loss_, (out, new_batch_stats)), grads = grad_fn(
            state.params,
            state.batch_stats,
            state,
            batch,
            static_args,
            *rngs,
            )

        new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
        
        metrics = self.cal_minibatch_metrics(out, batch, static_args, state)
        
        return new_state, metrics
    
    
    
    @classmethod
    @partial(jax.jit, static_argnames=["self", "static_args"])
    def cal_minibatch_metrics(self, out, batch, static_args, state):
        y_chip = out
        
        l_chip = optax.sigmoid_binary_cross_entropy(y_chip, batch['y_chip']).mean() * static_args['k_chip']
        
        metrics = {"l_chip": l_chip} 
        return metrics 
    
    @classmethod 
    @partial(jax.jit, static_argnames=["self", "static_args"])
    def apply_model(self, state, ds_batched, static_args) -> Tuple[jax.Array, ...]:
        '''
        apply model when not training, should be jitted
        Args:
            static_args : dict 
        Returns:
            model.apply(params, ds_batched[key1], ..., *args, False)    
        '''
        out = state.apply_fn({"params": state.params,
                              "batch_stats": state.batch_stats},
                          ds_batched['x_acc'],
                          False)
        return out 
    
    def print_epoch_metrics(self, epoch_metrics_np, t_start, t_stop):
        '''
        example:
            print('epoch: %d, loss: %.4f, time_elapsed %.0fs' % (
                self.epoch,
                epoch_metrics_np['loss'], 
                t_stop-t_start))
        '''
        print('epoch: %d, l_chip: %.4f, val_l_chip: %.4f, time_elapsed %.0fs' % (
                self.epoch,
                epoch_metrics_np['l_chip'],
                epoch_metrics_np['val_l_chip'],
                t_stop-t_start)
              )
        return
    
    def pred_y(self, n_batch=1, ds=None, static_args=flax.core.freeze({}), save_self=True, return_res=False):
        out_acc = np.zeros_like(ds['y_chip'])
        ds_size_acc = out_acc.shape[0]
        n_cyc = n_batch if ds_size_acc%n_batch==0 else n_batch+1
        
        perms = np.arange(n_batch*(ds_size_acc//n_batch)).reshape((n_batch, (ds_size_acc//n_batch)))
        for i in tqdm(range(n_cyc)):
            if i!= n_cyc-1:
                perm = perms[i]
            else:
                perm = np.arange(i*(ds_size_acc//n_batch), ds_size_acc)
            ds_batched = self.create_batch(perm=(perm, np.arange(1)), ds=ds)
            out_acc[perm, :] = self.apply_model( 
                                        self.state,
                                        ds_batched,
                                        static_args
                                        )
        
        if save_self:
            self.y_pred = {'chip': out_acc} 
        if return_res:
            return {'chip': out_acc} 
        else:
            return 
    
    
    
    
    
    
    





def loss(params, batch_stats, state, batch : dict, static_args : dict, *rngs):
    out, new_batch_stats = state.apply_fn({"params": params,
                                            "batch_stats": batch_stats},
                                  batch['x_acc'],
                                  True,
                                  rngs={'dropout': rngs[0], 'stc_rev_compl': rngs[1], 'stc_shift':rngs[2]}, 
                                  mutable=['batch_stats']
                                  )
    y_chip = out
    l_chip = optax.sigmoid_binary_cross_entropy(y_chip, batch['y_chip']).mean() * static_args['k_chip']
    l = l_chip
    return l, (out, new_batch_stats['batch_stats']) 
    
    
    