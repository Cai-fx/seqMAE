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


import numpy as np
import jax.numpy as jnp
from .base_model import BaseModel
from tqdm import tqdm
from typing import Callable
from flax.core.frozen_dict import FrozenDict



class scb_pretrain_shareseq(nn.Module):
    latent_dim : int = 32 
    infer_filter_num : int = 288 
    kernel_sizes : tuple = (17, 5, 5, 5, 5, 5, 5, 1)
    pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    activation_func : Callable = nn.gelu
    n_rng : int = 3
    
    @nn.compact
    def __call__(self, seq, atac_dpth, train):
        seq = Seq2PeakEmb_orig(latent_dim = self.latent_dim,
                                known_filter = False,
                                infer_filter_num = self.infer_filter_num,
                                kernel_sizes = self.kernel_sizes,
                                pool_sizes = self.pool_sizes,
                                layer_features  = self.layer_features,
                                activation_func = self.activation_func
                                )(seq, 0, train)                        #(n_peak, d)
        
        # rna = self.param("Z", nn.initializers.constant(0), (atac_dpth.shape[0], self.latent_dim))               #(n_cell, d)
        # seq = seq @ rna.T                                                   #(n_peak, n_cell)
        
        seq = MLP(features=(64, 256, atac_dpth.shape[0]), activation=nn.relu)(seq, train)
        # bs = self.param('bs', nn.initializers.constant(1), (1,))
        # bias = self.param('bias', nn.initializers.constant(0.), seq.shape[-1])
        # seq += bias 
        return seq


    
class Model(BaseModel):
    def __init__(self, 
                 preprocess_folder_acc = None,
                 preprocess_folder_rna = None,
                 atac = None,
                 model_config = {}, 
                 **kwargs):
        # rna: (n_cell, n_gene)
        self.preprocess_folder_acc = preprocess_folder_acc
        self.preprocess_folder_rna = preprocess_folder_rna
        self.atac = atac 
        self.atac_dpth = np.array(np.log10(np.sum(atac, axis=0))).flatten()       #(n_cell, )
        
        self.model_config = model_config
        super().__init__(**kwargs)
        return 

    def create_model(self):
        self.model = scb_pretrain_shareseq(**self.model_config)
        
        return
    
    def read_accrna_ds(self, ds_key='train', **kwargs) -> dict:
        train_ds_acc = self.read_ds(ds_key=ds_key, type="acc")
        # print(cell_ids.shape, self.atac_dpth.shape )
        ds = {'x_acc': train_ds_acc['x'], 
              'y_acc': train_ds_acc['y'],
              'atac_dpth': self.atac_dpth
              }
        return ds 
    
    def shuffle_batch(self, rng : Optional[PRNGKeyArray], ds : dict, num_batch : int) -> Tuple:
        ds_size_acc = ds['x_acc'].shape[0]              #n_peak
        
        perms_acc = self._shuffle_batch(rng, ds_size_acc, num_batch)        #(n_batch, batch_size)
        perms_rna = jnp.zeros((perms_acc.shape[0], 1))
        return perms_acc, perms_rna
    
    def create_batch(self, perm, ds: dict) -> dict:
        perm_acc, _ = perm
        batch = {'x_acc': ds['x_acc'][perm_acc,:],
                'y_acc': ds['y_acc'][perm_acc,:],
                'atac_dpth': ds['atac_dpth']
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
        y_acc = out
        
        l_acc = optax.sigmoid_binary_cross_entropy(y_acc, batch['y_acc']).mean() * static_args['k_acc']
        
        metrics = {"l_acc": l_acc} 
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
                          ds_batched['atac_dpth'],
                          False)
        return out 
    
    def print_epoch_metrics(self, epoch_metrics_np, t_start, t_stop):
        print('epoch: %d, l_acc: %.4f, val_l_acc: %.4f, time_elapsed %.0fs' % (
                self.epoch,
                epoch_metrics_np['l_acc'],
                epoch_metrics_np['val_l_acc'],
                # self.state.params['bs'],
                # self.state.params['bs_intercept'],
                t_stop-t_start)
              )
        return
    
    def pred_y(self, n_batch=1, ds=None, static_args=flax.core.freeze({}), save_self=True, return_res=False):
        out_acc = np.zeros_like(ds['y_acc'])
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
            self.y_pred = {'acc': out_acc} 
        if return_res:
            return {'acc': out_acc} 
        else:
            return 
    
    
    
    
    
    
    





def loss(params, batch_stats, state, batch : dict, static_args : dict, *rngs):
    out, new_batch_stats = state.apply_fn({"params": params,
                                            "batch_stats": batch_stats},
                                  batch['x_acc'],
                                  batch['atac_dpth'],
                                  True,
                                  rngs={'dropout': rngs[0], 'stc_rev_compl': rngs[1], 'stc_shift':rngs[2]}, 
                                  mutable=['batch_stats']
                                  )
    y_acc = out
    l_acc = optax.sigmoid_binary_cross_entropy(y_acc, batch['y_acc']).mean() * static_args['k_acc']
    l = l_acc
    return l, (out, new_batch_stats['batch_stats']) 
    
    
    