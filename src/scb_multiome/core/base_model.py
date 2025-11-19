from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod
import flax.training
import flax.training.early_stopping
from jax._src.prng import PRNGKeyArray 

import numpy as np
import jax 
import optax
from functools import partial
from tqdm import tqdm
from ..utils.IOs import seq_acc_generator, seq_rna_generator, load_params
from ..utils.train_utils import trainState
import flax 

import h5py 
from time import perf_counter
import pickle


class BaseModel(ABC):
    def __init__(self, **kwargs):  
        for key in kwargs:
            setattr(self, key, kwargs[key])
        pass

    @abstractmethod
    def create_model(self):
        """
        init self.model
        """
        return

    def create_train_state(self, 
                           *init_args,
                           learning_rate=1e-2,
                           pretrained_path=None, 
                           load_keys=["params_all"],
                           exclude_list=[],
                           clip_grad=False):
        '''clip_grad: False or float'''
        
        init_rng=jax.random.PRNGKey(0)
            
        @jax.jit
        def _init_model(init_rng):
            return self.model.init(init_rng, *init_args, False)
        
        init_params = _init_model(init_rng)
        if not (pretrained_path is None):
            print("initializing with pretrained params...")
            init_params = load_params(init_params, pretrained_path, load_keys=load_keys, exclude_list=exclude_list)
        print(jax.tree_util.tree_map(lambda x: x.shape, init_params))
        
        if clip_grad:
            tx = optax.chain(
                optax.clip_by_global_norm(clip_grad),
                optax.adam(learning_rate),
                )
        else:
            tx = optax.adam(learning_rate=learning_rate)
        
        self.state = trainState.create(apply_fn = self.model.apply,
                                        params = init_params['params'],
                                        batch_stats = init_params['batch_stats'],
                                        tx = tx)
        return
    
    
    
    ## reading data ######################################################################
    @abstractmethod
    def read_accrna_ds(self, ds_key='train', **kwargs) -> dict:
        '''
        wraps around self.read_ds 
        returns dict
        '''
        return 
    
    def read_ds(self, ds_key='train', type="acc"):
        if type == 'acc':
            preprocess_folder = self.preprocess_folder_acc
            m = self.atac.tocsr()       ## sparse, binary data
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
        
        m_train = m[train_ids,:]
        
        if type == 'acc':
            if isinstance(m, np.ndarray):                 ##non binary, non sparse atac data
                train_ds = seq_rna_generator(train_data, m_train)()
            else:
                train_ds = seq_acc_generator(train_data, m_train)()
        if type == 'rna':
            train_ds = seq_rna_generator(train_data, m_train)()

        ds = np.array([_x for _x in train_ds], dtype=object)
        train_x = np.array(list(zip(*ds))[0])
        train_y = np.array(list(zip(*ds))[1])
        # print(train_x.shape, train_y.shape)
        if (type=="acc"):
            return {'x': train_x, 'y': train_y} 
        else:
            return {'x': train_x, 'y_u': self.rna_u[train_ids,:], 'y_s': self.rna_s[train_ids,:]}
    

    ## batching data ######################################################################
    @classmethod
    def _shuffle_batch(self, rng, ds_size, num_batch):
        '''
        skips incompletely batches 
        '''
        batch_size = ds_size//num_batch
        if rng is None:
            perms=np.arange(ds_size)
        else: 
            perms=jax.random.permutation(rng, ds_size)
        perms=perms[:num_batch * batch_size]        # skip incomplete batch
        perms=perms.reshape((num_batch, batch_size))
        return perms 
    
    @abstractmethod
    def shuffle_batch(self, rng : Optional[PRNGKeyArray], ds, num_batch) -> Tuple:
        """wraps around _shuffle_batch

        Args:
            rng (Optional[PRNGKeyArray]): if rng is None, no shuffle 
            ds (_type_): dictionary
            num_batch (_type_): _description_

        Returns:
            Tuple: tuple of array of perm indices, (num_batch, batch_size)
        """
        return
    
    @abstractmethod
    def create_batch(self, perm, ds: dict) -> dict: 
        """takes ds, permutate, output batch (dict)
        perms : array of ints, size (batch_size, ) :Tuple[np.ndarray[int], ...]

        Args:
            perm (_type_): _description_
            ds (dict): _description_

        Returns:
            dict: example 
            {'x_acc': ds['x_acc'][perm_acc,:], 'x_rna': ds['x_rna'][perm_rna,:], 
            'y_acc': ds['y_acc'][perm_acc,:],
            'y_rna_u': ds['y_rna_u'][perm_rna,:], 'y_rna_s': ds['y_rna_s'][perm_rna,:]}
        """
        return
    
    
    ## train ######################################################################
    def fit(self, 
            train_ds : dict,
            val_ds : dict,
            static_args : flax.core.FrozenDict = flax.core.freeze({}),
            n_epochs : int = 200,
            num_train_batch : int = 100,
            num_val_batch :int = 10,
            rng = jax.random.PRNGKey(3456),
            tidy_metrics = True,
            early_stop = False,
            early_stop_kwargs = {"stop_metric":"", "min_delta":1e-4, "patience":3}, 
            ):
        """wraps around self.train_epoch

        Args:
            train_ds (dict): _description_
            val_ds (dict): _description_
            static_args (flax.core.FrozenDict, optional): _description_. Defaults to flax.core.freeze({}).
            n_epochs (int, optional): _description_. Defaults to 200.
            num_train_batch (int, optional): _description_. Defaults to 100.
            num_val_batch (int, optional): _description_. Defaults to 10.
            rng (_type_, optional): _description_. Defaults to jax.random.PRNGKey(3456).
            tidy_metrics (bool, optional): _description_. Defaults to True.
            early_stop (bool, optional): _description_. Defaults to False.
            early_stop_kwargs (dict, optional): _description_. Defaults to {"stop_metric":"", "min_delta":1e-4, "patience":3}.
        """
        if "metrics" not in self.__dict__:
            self.metrics=[] 
        if early_stop:
            early_stopper = flax.training.early_stopping.EarlyStopping(min_delta=early_stop_kwargs["min_delta"],
                                                                       patience=early_stop_kwargs["patience"]) 
        self.epoch = 0
        for i in range(n_epochs):
            rng, input_rng1 = jax.random.split(rng, 2)  
            self.train_epoch(train_ds=train_ds,
                            val_ds=val_ds,
                            rng=input_rng1,
                            n_train_batch=num_train_batch,
                            n_val_batch=num_val_batch,
                            static_args=static_args,
                            )
            self.epoch += 1
            if early_stop:
                _has_improved, early_stopper = early_stopper.update(self.metrics[-1][early_stop_kwargs["stop_metric"]])
                if early_stopper.should_stop:
                    print(f'Met early stopping criteria, breaking')
                    break
        
        if tidy_metrics:
            self.metrics = {key:np.array([self.metrics[i][key] for i in range(len(self.metrics))]) 
                            for key in self.metrics[0].keys()}
        return 
    
    def train_epoch(self, 
                         train_ds : dict,
                         val_ds : dict,
                         rng : PRNGKeyArray, 
                         n_train_batch : int,
                         n_val_batch : int,
                         static_args : flax.core.FrozenDict
                         ) -> None:
        """update self.state and self.metrics

        Args:
            train_ds (dict): _description_
            val_ds (dict): _description_
            rng (PRNGKeyArray): _description_
            n_train_batch (int): _description_
            n_val_batch (int): _description_
            static_args (flax.core.FrozenDict): _description_
        """
        
        t_start = perf_counter()
        
        rng, shuffle_rng, rngs = jax.random.split(rng, 3)
        
        perms = self.shuffle_batch(shuffle_rng, ds=train_ds, num_batch=n_train_batch)
        batch_metrics = np.empty(n_train_batch, dtype=dict)   
        rngs = (rngs, )      
        for i, perm in enumerate(tqdm(zip(*perms), desc=f'epoch', unit="batch", ascii=' >=', total=n_train_batch)):   
            rngs = jax.random.split(rngs[0], num=self.model.n_rng+1)
            batch = self.create_batch(perm=perm, ds=train_ds)
            self.state, metrics = self.train_step_func(self.state,
                                                       batch,
                                                       static_args,
                                                       *rngs[1:])
            batch_metrics[i] = metrics

        epoch_metrics_np = self.mean_batch_metric(batch_metrics)
        val_metrics = self.validation_metrics(val_ds=val_ds, 
                                              n_batch=n_val_batch,
                                              static_args=static_args)

        t_stop = perf_counter()
        
        metrics = dict(epoch_metrics_np, **val_metrics)
        self.metrics.append(metrics)
        self.print_epoch_metrics(metrics, t_start=t_start, t_stop=t_stop)
        return
    
    @classmethod
    @abstractmethod
    def train_step_func(self, 
                        state : trainState, 
                        batch : dict, 
                        static_args : dict, 
                        *rngs) -> Tuple[trainState, dict]:
        '''
        gradient descend for each minibatch, jitted function
         
        Returns:
            new_state, metrics     
        '''
        return
    
    
    
    
    ### metrics ######################################################################
    def validation_metrics(self, 
                           val_ds : dict,
                           n_batch : int,
                           static_args : dict):
        batch_metrics = np.empty(n_batch, dtype=dict)
        perms=self.shuffle_batch(rng=None, ds=val_ds, num_batch=n_batch)
        for i, perm in enumerate(zip(*perms)):
            val_ds_batched = self.create_batch(perm=perm, ds=val_ds)
            out = self.apply_model(self.state,
                                    val_ds_batched,
                                    static_args
                                    )
            val_metrics = self.cal_minibatch_metrics(out, val_ds_batched, static_args, self.state)
            batch_metrics[i] = val_metrics
        epoch_metrics_np = self.mean_batch_metric(batch_metrics)
        epoch_metrics_np = {f'val_{k}':epoch_metrics_np[k] for k in epoch_metrics_np}
        return epoch_metrics_np
        
    @classmethod
    @abstractmethod
    def cal_minibatch_metrics(self, out, batch, static_args) -> dict:
        '''
        example: {"loss": loss, ...}
        '''
        return

    @classmethod
    def mean_batch_metric(self, batch_metrics):
        '''
        Args:
            batch_metrics: np.ndarray[dict]
        '''
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]}
        return epoch_metrics_np

    @abstractmethod
    def print_epoch_metrics(self, epoch_metrics_np, t_start, t_stop):
        '''
        example:
            print('epoch: %d, loss: %.4f, time_elapsed %.0fs' % (
                self.epoch,
                epoch_metrics_np['loss'], 
                t_stop-t_start))
        '''
        return
                   
        
    @abstractmethod
    def pred_y(self, n_batch=100, ds=None, static_args={}):
        return
    
    ## saving & loading results 
    def save_results(self, save_path=None, 
                     attr_keys=['params_all', 'y_pred', 'metrics']):
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
                  keys : list =['params_all', 'y_pred', 'metrics']):
        if 'params_all' in keys:
            self.state = trainState.create(apply_fn = self.model.apply,
                                            params=data['params_all']['params'],
                                            batch_stats = data['params_all']['batch_stats'],
                                            tx=optax.adam(learning_rate=1e-2))
            keys.remove('params_all')
        for key in keys:
            setattr(self, key, data[key])
        return
    

    @classmethod
    @abstractmethod
    def apply_model(self, state, ds_batched, static_args) -> Tuple[jax.Array, ...]:
        '''
        apply model when not training, should be jitted
        Args:
            static_args : dict 
        Returns:
            model.apply(params, ds_batched[key1], ..., *args, False)    
        '''
        return

    
    
    