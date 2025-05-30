from typing import Any
import numpy as np
import jax 
import flax.linen as nn
import flax
import optax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
from ..utils.scb_blocks import Decoder, conv_block, conv_tower, dense_block, Stc_rev_compl, Stc_shift
from ..utils.IOs import seq_rna_generator, seq_acc_generator, load_params
from ..utils.eval_metrics import compute_metrics_acc
from ..utils.train_utils import *

import h5py 
import scipy.sparse as sparse
from time import perf_counter
import pickle


from jax import config
config.update("jax_debug_nans", True)
# config.update('jax_array', False)



class scb_model(nn.Module):
    n_cell : int
    peakemb_latent_dim : int = 32
    
    @nn.compact
    def __call__(self, x, train):           #(n, 1344, 4) #(n_cell, n_gene)
        x = Stc_rev_compl()(x, deterministic= not train)
        x = Stc_shift()(x, deterministic= not train)
        x = conv_block(features=288,
                        kernel_size=(17,),
                        pool_size=(3,))(x, train)
        x = conv_tower(kernel_size=(5,),
                        pool_size=(2,))(x, train)
        x = conv_block(features = 256,
                        kernel_size=(1,),
                        pool_size=(1,))(x, train)
        x = dense_block(features=self.peakemb_latent_dim)(x, train)
        x = nn.gelu(x)
        Z = self.param('Z', self.init_cellemb, (self.n_cell, self.peakemb_latent_dim))
        x = Decoder(self.n_cell)(x, Z)
        return x                    
    
    @staticmethod
    def init_cellemb(rng, shape):
        return flax.linen.initializers.lecun_normal()(rng, shape)


class CNN(nn.Module):
    peakemb_latent_dim : int = 32
    @nn.compact
    def __call__(self, x, train):           #(n, 1344, 4) #(n_cell, n_gene)
        x = Stc_rev_compl()(x, deterministic= not train)
        x = Stc_shift()(x, deterministic= not train)
        x = conv_block(features=288,
                        kernel_size=(17,),
                        pool_size=(3,))(x, train)
        x = conv_tower(kernel_size=(5,),
                        pool_size=(2,))(x, train)
        x = conv_block(features = 256,
                        kernel_size=(1,),
                        pool_size=(1,))(x, train)
        x = dense_block(features=self.peakemb_latent_dim)(x, train)
        x = nn.gelu(x)
        return x




#################################



class scb_orig:
    def __init__(self,
                 atac,
                 preprocess_folder_rna=None,
                 preprocess_folder_acc=None,
                 peakemb_latent_dim=32,
                 
                 k_acc_c = 1.,
                 ) -> None:
        
        self.atac = atac

        self.preprocess_folder_rna = preprocess_folder_rna
        self.preprocess_folder_acc = preprocess_folder_acc
        self.n_cell = atac.shape[1]
        self.peakemb_latent_dim = peakemb_latent_dim
        self.k_acc_c = k_acc_c
        
        return
    
    def create_model(self):
        self.model = scb_model(n_cell=self.n_cell, 
                                peakemb_latent_dim=self.peakemb_latent_dim                        
                                )
        
        return
    
    def create_train_state(self, learning_rate=1e-2, init_rng=jax.random.PRNGKey(0), print_shape=True,
                           pretrained_path=None, load_keys=["params_all"],
                           freeze_cnn = False):
            
        @jax.jit
        def _init_model(init_rng):
            return self.model.init(init_rng, np.zeros((3, 1344, 4)), False)
        
        init_params = _init_model(init_rng)
        
        if not (pretrained_path is None):
            init_params = load_params(init_params, pretrained_path, load_keys=load_keys)
        
        if print_shape:
            print(jax.tree_util.tree_map(lambda x: x.shape, init_params))
        
        tx = optax.adam(learning_rate=learning_rate)
        if freeze_cnn:
            mask = process_nested_dict(init_params, "Seq2PeakEmb_0")
            tx = optax.multi_transform({True: tx, False: zero_grads()},
                                        mask['params'])
            # print(mask)
    
        self.state = trainState.create(apply_fn = self.model.apply,
                                        params = init_params['params'],
                                        batch_stats = init_params['batch_stats'],
                                        tx = tx)
        return

    
    def read_accrna_train_ds(self, ds_key='train', save_self=True, shuffle_input_seq=False):        
        train_ds_acc = self.read_train_ds(ds_key=ds_key, type='acc', shuffle_input_seq=shuffle_input_seq)
        
        ds = {'x_acc': train_ds_acc['x'],
              'y_acc': train_ds_acc['y']}
        
        if save_self:
            self.train_ds = ds
            
        return ds
    
    
    def read_train_ds(self, ds_key='train', type="acc", shuffle_input_seq=False):
        if type == 'acc':
            preprocess_folder = self.preprocess_folder_acc
            m = self.atac.tocsr()       ## (n_peak, n_cell)
            train_data = '%s/%s_seqs.h5'%(preprocess_folder, ds_key)
            split_file = '%s/splits.h5'%preprocess_folder
        if type == 'rna':
            preprocess_folder = self.preprocess_folder_rna
            m = self.rna_u
            train_data = '%s/tss_%s_seqs.h5'%(preprocess_folder, ds_key) if ds_key != 'all' else '%s/tss_seqs.h5'%(preprocess_folder)
            split_file = '%s/gene_splits.h5'%preprocess_folder
        
        if ds_key == 'all':
            train_ids = np.arange(m.shape[0])
        else:    
            with h5py.File(split_file, 'r') as hf:
                train_ids = hf[f'{ds_key}_ids'][:]
        
        if shuffle_input_seq:
            shuffle_input_idx = self.shuffle_seq_idx(n_gene=self.atac.shape[0])
            shuffle_input_idx = shuffle_input_idx[train_ids, :]
        else:
            shuffle_input_idx = np.tile(np.arange(1344, dtype=np.uint16), (len(train_ids), 1))    
        
        m_train = m[train_ids,:]
        
        if type == 'acc':
            train_ds = seq_acc_generator(train_data, m_train)()
        if type == 'rna':
            train_ds = seq_rna_generator(train_data, m_train)()

        ds = np.array([_x for _x in train_ds], dtype=object)
        train_x = np.array(list(zip(*ds))[0])
        train_y = np.array(list(zip(*ds))[1])
        
        train_x = jax.vmap(lambda a, b : a[b, :], in_axes=(0,0), out_axes=0)(train_x, shuffle_input_idx)
        
        if (type=="acc"):
            return {'x': train_x, 'y': train_y} 
        else:
            return {'x': train_x, 'y_u': self.rna_u[train_ids,:], 'y_s': self.rna_s[train_ids,:]}
    
    @classmethod
    def shuffle_seq_idx(cls, n_gene, 
                        rng=jax.random.PRNGKey(7621),
                        ):
        _rngs = jax.random.split(rng, n_gene)
        shuffle_input_idx = jax.vmap(lambda _r: jax.random.permutation(_r, np.arange(1344, dtype=np.uint16)), 
                                            in_axes=0, out_axes=0)(_rngs)
        return shuffle_input_idx
    
    
    
    def print_epoch_metrics(self, epoch_metrics_np, t_start, t_stop):
        print('epoch: %d, loss: %.4f, c: %.3f, time_elapsed %.0fs' % (
            self.epoch,
            epoch_metrics_np['loss'], 
            epoch_metrics_np['acc_c_loss'], 
            t_stop-t_start))
        return
    
    def shuffle_batch(self, rng):
        rng1, rng2 = jax.random.split(rng, 2)
        perms_acc = jax.random.permutation(rng1, self.ds_size_acc)
        perms_acc = perms_acc[:self.num_batch * self.batch_size_acc]       # skip incomplete batch
        perms_acc = perms_acc.reshape((self.num_batch, self.batch_size_acc))
        return perms_acc 
    
    
    def train_cycle_step(self, rng, shuffle_seq=False):            
        t_start = perf_counter()
        
        rng, shuffle_rng, new_dropout_rng = jax.random.split(rng, 3)
        perms_acc = self.shuffle_batch(shuffle_rng)
        batch_metrics = np.empty(self.num_batch, dtype=dict) 
                
        for i, perm_acc in enumerate(tqdm(perms_acc, desc=f'epoch', unit="batch", ascii=' >=', total=self.num_batch)):   
            dropout_rng, stc_rev_compl_key, stc_shift_key, new_dropout_rng = jax.random.split(new_dropout_rng, num=4)
            batch = {'x_acc': self.train_ds['x_acc'][perm_acc,:], 
                     'y_acc': self.train_ds['y_acc'][perm_acc,:],
                     }
            self.state, metrics = train_step(self.state, 
                                            self.model, 
                                            batch, 
                                            dropout_rng, 
                                            stc_rev_compl_key,
                                            stc_shift_key,
                                            
                                            k_acc_c=self.k_acc_c,
                                            shuffle_seq = shuffle_seq
                                            )
            batch_metrics[i] = metrics

        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]}
        t_stop = perf_counter()
        self.print_epoch_metrics(epoch_metrics_np, t_start=t_start, t_stop=t_stop)
        self.metrics.append(epoch_metrics_np)
        return
    
    
    
    def fit(self, 
            n_epochs = 200, 
            num_batch=100,
            rng = jax.random.PRNGKey(3456),
            shuffle_seq = False,
            ):
        self.metrics=[]
        self.num_batch = num_batch
        self.ds_size_acc = self.train_ds['x_acc'].shape[0]
        self.batch_size_acc = self.ds_size_acc // num_batch
        
        self.epoch = 0
        
        for i in range(n_epochs):
            rng, input_rng1 = jax.random.split(rng, 2)  
            self.train_cycle_step(input_rng1, shuffle_seq=shuffle_seq)
            self.epoch += 1
        
        self.metrics = {key:np.array([self.metrics[i][key] for i in range(len(self.metrics))]) 
                        for key in self.metrics[0].keys()}
        return 
    
    
    def pred_y(self, batch_size=256, ds=None):
        if ds is None:
            ds = self.train_ds

        batch_num_acc = np.ceil(ds['x_acc'].shape[0]/batch_size).astype('int')
        C = np.zeros(ds['y_acc'].shape)
        
        
        for i in tqdm(range(batch_num_acc), ascii=' >=', desc='predict acc '):
            if i!=batch_num_acc-1:
                idx = np.arange(i*batch_size, (i+1)*batch_size)
            else:
                idx = np.arange(i*batch_size, ds['x_acc'].shape[0])
            x_acc= ds['x_acc'][idx, :]
            C[idx,:] = apply_model(self.model,
                                    {'params': self.state.params, 'batch_stats': self.state.batch_stats},
                                    x_acc
                                    )
            
        self.y_pred = {'C': C}
        return

    def get_peak_emb(self, batch_size=256, ds=None, type='acc'):
        cnn = CNN(peakemb_latent_dim=self.peakemb_latent_dim)
        
        @jax.jit
        def apply_cnn(params, x):
            return cnn.apply(params, x, False)
        
        if ds is None:
            ds = self.train_ds
        
        batch_num = np.ceil(ds[f'x_{type}'].shape[0]/batch_size).astype('int')
        y_pred = np.zeros((ds[f'x_{type}'].shape[0], self.peakemb_latent_dim))
        
        for i in tqdm(range(batch_num)):
            if i!=batch_num-1:
                idx = np.arange(i*batch_size, (i+1)*batch_size)
            else:
                idx = np.arange(i*batch_size, ds[f'x_{type}'].shape[0])
            y_pred[idx,:] = apply_cnn({'params': self.state.params, 'batch_stats': self.state.batch_stats},
                                                ds[f'x_{type}'][idx, :])
        if type == 'acc':
            self.peak_emb_acc = y_pred                      #(n_peak, 32)
        return 
    
    
    def save_results(self, save_path=None, attr_keys=['params_all', 'y_pred', 'metrics']):
        if 'params_all' in attr_keys:
            params_all = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
            d_save = {'params_all':params_all}
            attr_keys.remove('params_all')
        else:
            d_save = {}

        d_save.update({attr_key : getattr(self, attr_key) for attr_key in attr_keys})
        with open(save_path, "wb") as handle:
            pickle.dump(d_save,
                        handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)
        return 

    
    def load_data(self, data:dict, 
                  keys=['params_all', 'y_pred', 'metrics', 
                        'peak_emb_acc', 'peak_emb_rna',
                        'cell_emb_acc', 'cell_emb_rna', 
                        'gs_tau','k']):
        if 'params_all' in keys:
            self.state = trainState.create(apply_fn = self.model.apply,
                                            params=data['params_all']['params'],
                                            batch_stats = data['params_all']['batch_stats'],
                                            tx=optax.adam(learning_rate=1e-2))
            keys.remove('params_all')

        for key in keys:
            setattr(self, key, data[key])
        return
    




def acc_loss(out_acc, batch, k_acc):
    return k_acc * optax.sigmoid_binary_cross_entropy(logits=out_acc, labels=batch['y_acc']).mean()




def loss(params : flax.core.FrozenDict,
         model : scb_model,
         batch : dict, 
         state : trainState,  
         dropout_key, #: jax.random.PRNGKeyArray, 
         stc_rev_compl_key, #: jax.random.PRNGKeyArray,
         stc_shift_key, #: jax.random.PRNGKeyArray,
         k_acc_c : float,
         shuffle_seq : bool = False):
    
    if shuffle_seq:
        dropout_key, seq_key = jax.random.split(dropout_key)
        batch['x_acc'] = jax.random.permutation(seq_key, batch['x_acc'], axis=1)
    
    C, new_batch_stats = model.apply(
            {'params': params, 
             'batch_stats': state.batch_stats}, 
            batch['x_acc'],
            True,
            rngs={'dropout': dropout_key, 'stc_rev_compl': stc_rev_compl_key, 'stc_shift': stc_shift_key}, 
            mutable=['batch_stats']) 
        
    loss = acc_loss(C, batch, k_acc=k_acc_c)
            
    return loss, (C, new_batch_stats['batch_stats'])





@partial(jax.jit, static_argnums=(1,7))
def train_step(state, 
                model, 
                batch, 
                dropout_rng, 
                stc_rev_compl_key,
                stc_shift_key,
                k_acc_c,
                shuffle_seq = False):
    
    grad_fn = jax.value_and_grad(loss, has_aux=True, argnums=0) 
    (loss_, (C, new_batch_stats)), grads = grad_fn(
        state.params, 
        model, 
        batch, 
        state, 
        dropout_rng, 
        stc_rev_compl_key,
        stc_shift_key,
        k_acc_c,
        
        shuffle_seq = shuffle_seq
        )

    new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    
    metrics_acc_c = compute_metrics_acc(y_pred=C, labels=batch['y_acc'], key='acc_c')
    
    
    metrics_acc_c['acc_c_loss'] *= k_acc_c
    
    metrics = metrics_acc_c
    
    metrics['loss'] = loss_
    return new_state, metrics
    

@partial(jax.jit, static_argnums=0)
def apply_model(model, params, x_acc):
    return model.apply(params, x_acc, False)          # not train



