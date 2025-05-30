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
from ..utils.eval_metrics import compute_metrics_rna, compute_metrics_acc
from tqdm import tqdm
from typing import Callable
from flax.core.frozen_dict import FrozenDict
import logging 



class scb_TF_chip(nn.Module):
    latent_dim : int = 32 
    infer_filter_num : int = 288 
    kernel_sizes : tuple = (17, 5, 5, 5, 5, 5, 5, 1)
    pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    activation_func : Callable = nn.gelu
    chip_dim : int = 1
    
    encoder_features : tuple = (64, 32) 
    decoder_features : tuple = (64, )
    mlp_activation : Callable = nn.relu         #(used to be softplus, not in any usable result)
    mlp_batchnorm : bool = False
    n_rng : int = 3 
    
    @nn.compact
    def __call__(self, seq, rna, atac_dpth, train, mlp_only=False, fixbias=True):
        n_gene = rna.shape[-1]
        seq = Seq2PeakEmb_orig(latent_dim = self.latent_dim,
                                known_filter = False,
                                infer_filter_num = self.infer_filter_num,
                                kernel_sizes = self.kernel_sizes,
                                pool_sizes = self.pool_sizes,
                                layer_features  = self.layer_features,
                                activation_func = self.activation_func
                                )(seq, 0, train)                        #(n_peak, d)
        
        if mlp_only:
            seq = jax.lax.stop_gradient(seq)
        
        # chip = nn.Dropout(rate=0.5)(seq, deterministic=not train)
        # chip = nn.Dense(features=self.chip_dim,
        #                 use_bias=True)(chip)                                 #(n_peak, 1)
        chip = MLP(name="mlp_chip",
                   features=(self.chip_dim, ),
                   batchnormalize=True,
                   use_bias=True,
                   dropout_rate=0.3)(seq, train)
        
        # print(chip.shape)
        rna = MLP(features=self.encoder_features,
                  activation=self.mlp_activation,
                  batchnormalize=self.mlp_batchnorm)(rna, train)               #(n_cell, d)
        seq = seq @ rna.T                                                   #(n_peak, n_cell)
        
        
        bs = self.param('bs', nn.initializers.constant(1), (1,))
        intercept = self.param('bs_intercept', nn.initializers.constant(-10), (1,))
            
        
        bias = bs * atac_dpth + intercept                          #(n_cell, )
                
        if mlp_only and fixbias:
            bias = jax.lax.stop_gradient(bias)
        seq += bias 
        
        rna_out = MLP(features=(*self.decoder_features, n_gene),
                      activation=self.mlp_activation)(rna, train)
        return seq, rna_out, chip


    
class Model(BaseModel):
    def __init__(self, 
                 preprocess_folder_acc = None,
                 preprocess_folder_rna = None,
                 atac = None,
                 rna = None,
                 chip_bulk = None,                   
                 model_config = {}, 
                 **kwargs):
        '''
        rna: (n_cell, n_gene)
        chip: (n_peak, n_tf_ref)
        '''
        self.preprocess_folder_acc = preprocess_folder_acc
        self.preprocess_folder_rna = preprocess_folder_rna
        self.atac = atac 
        self.rna = rna 
        self.chip_bulk =  chip_bulk 
        self.atac_dpth = np.array(np.log10(np.sum(atac, axis=0))).flatten()       #(n_cell, )
        
        self.model_config = model_config
        super().__init__(**kwargs)
        return 

    def create_model(self):
        self.model = scb_TF_chip(chip_dim=self.chip_bulk.shape[1], **self.model_config)
        
        return
    
    def read_cell_split(self, ds_key='train') -> dict:
        if ds_key == 'all':
            cell_ids = np.arange(self.rna.shape[0])
        else:    
            split_file = f"{self.preprocess_folder_rna}/cell_splits.h5"
            with h5py.File(split_file, 'r') as hf:
                cell_ids = hf[f'{ds_key}_ids'][:]
        return cell_ids
    
    def read_peak_split(self, ds_key='train') -> dict:
        if ds_key == 'all':
            peak_ids = np.arange(self.atac.shape[0])
        else:
            split_file = f"{self.preprocess_folder_acc}/splits.h5"
            with h5py.File(split_file, 'r') as hf:
                peak_ids = hf[f'{ds_key}_ids'][:]
        return peak_ids
    
    def read_chip_ds(self, ds_key='train'):
        peak_ids = self.read_peak_split(ds_key=ds_key)
        return {"chip": self.chip_bulk[peak_ids]}
    
    def read_accrna_ds(self, ds_key='train', **kwargs) -> dict:
        train_ds_acc = self.read_ds(ds_key=ds_key, type="acc")
        cell_ids = self.read_cell_split(ds_key=ds_key)
        chip_ds = self.read_chip_ds(ds_key=ds_key)
        # print(cell_ids.shape, self.atac_dpth.shape )
        ds = {'x_acc': train_ds_acc['x'], 'rna': self.rna[cell_ids, :], 
              'y_acc': train_ds_acc['y'][:, cell_ids],
              'y_chip': chip_ds['chip'],
              'atac_dpth': self.atac_dpth[cell_ids],
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
                'rna': ds['rna'], 
                'y_acc': ds['y_acc'][perm_acc,:],
                'y_chip': ds['y_chip'][perm_acc,:],
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
        y_acc, y_rna, y_chip = out[0], out[1], out[2]
        
        l_acc = optax.sigmoid_binary_cross_entropy(y_acc, batch['y_acc']).mean() * static_args['k_acc']
        l_rna = optax.l2_loss(y_rna, batch['rna']).mean() * static_args['k_rna']
        l_chip = optax.sigmoid_binary_cross_entropy(y_chip, batch['y_chip']).mean() * static_args['k_chip']
        l_reg = (l1_loss(params=state.params['MLP_0']) + l1_loss(params=state.params['MLP_1'])) * static_args['k_reg']
                
        metrics = {"l_acc": l_acc,
                   "l_rna": l_rna,
                   "l_reg": l_reg,
                   "l_chip": l_chip,
                   "l": l_acc+l_rna+l_reg+l_chip} 
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
                          ds_batched['rna'],
                          ds_batched['atac_dpth'],
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
        print('epoch: %d, l: %.4f, l_val: %.4f, l_acc: %.4f, l_rna: %.4f, l_chip: %.4f, val_l_acc: %.4f, val_l_rna: %.4f, val_l_chip: %.4f, time_elapsed %.0fs' % (
                self.epoch,
                epoch_metrics_np['l'], 
                epoch_metrics_np['val_l'],
                epoch_metrics_np['l_acc'],
                epoch_metrics_np['l_rna'],
                epoch_metrics_np['l_chip'],
                epoch_metrics_np['val_l_acc'],
                epoch_metrics_np['val_l_rna'],
                epoch_metrics_np['val_l_chip'],
                t_stop-t_start)
              )
        return
    
    def pred_y(self, n_batch=1, ds=None, static_args=flax.core.freeze({}), save_self=True, return_res=False):
        out_acc, out_rna, out_chip = np.zeros_like(ds['y_acc']), np.zeros_like(ds['rna']), np.zeros_like(ds['y_chip'])
        ds_size_acc = out_acc.shape[0]
        n_cyc = n_batch if ds_size_acc%n_batch==0 else n_batch+1
        
        perms = np.arange(n_batch*(ds_size_acc//n_batch)).reshape((n_batch, (ds_size_acc//n_batch)))
        for i in tqdm(range(n_cyc)):
            if i!= n_cyc-1:
                perm = perms[i]

            else:
                perm = np.arange(i*(ds_size_acc//n_batch), ds_size_acc)
            ds_batched = self.create_batch(perm=(perm, np.arange(1)), ds=ds)
            out_acc[perm, :], out_rna, out_chip[perm, :] = self.apply_model( 
                                                            self.state,
                                                            ds_batched,
                                                            static_args
                                                            )
        
        if save_self:
            self.y_pred = {'acc': out_acc, 'rna': out_rna, 'chip':out_chip} 
        if return_res:
            return {'acc': out_acc, 'rna': out_rna, 'chip':out_chip} 
        else:
            return 
    
    
    
    
    
    
    





def loss(params, batch_stats, state, batch : dict, static_args : dict, *rngs):
    out, new_batch_stats = state.apply_fn({"params": params,
                                            "batch_stats": batch_stats},
                                  batch['x_acc'],
                                  batch['rna'],
                                  batch['atac_dpth'],
                                  True,
                                  static_args["mlp_only"],
                                  rngs={'dropout': rngs[0], 'stc_rev_compl': rngs[1], 'stc_shift':rngs[2]}, 
                                  mutable=['batch_stats']
                                  )
    y_acc, y_rna, y_chip = out[0], out[1], out[2]
    l_acc = optax.sigmoid_binary_cross_entropy(y_acc, batch['y_acc']).mean() * static_args['k_acc']
    l_rna = optax.l2_loss(y_rna, batch['rna']).mean() * static_args['k_rna']
    l_chip = optax.sigmoid_binary_cross_entropy(y_chip, batch['y_chip']).mean() * static_args['k_chip']
    l_reg = (l1_loss(params=params['MLP_0']) + l1_loss(params=params['MLP_1'])) * static_args['k_reg']
    l = l_acc + l_rna + l_reg + l_chip
    return l, (out, new_batch_stats['batch_stats']) 


def l1_loss(params:FrozenDict):
    ''' 
    {MLP_0: {Dense_0: {kernel: ...}}, MLP_1: {Dense_1: {kernel: ...}}}
    '''
    l=0
    for k, v in params.items():
        if 'Dense_' in k:
            l += sum(jnp.sum(jnp.abs(w)) for w in jax.tree_util.tree_leaves(params[k]['kernel']))       # used to be mean
    return l 
    
    