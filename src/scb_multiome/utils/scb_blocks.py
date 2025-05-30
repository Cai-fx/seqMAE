import os
import numpy as np
import jax
import jax.numpy as jnp            
from flax import linen as nn 
import numpy as np

from scipy import sparse
from typing import Any, Optional, Callable

import flax




'''model blocks'''

class conv_block(nn.Module):
    features: int
    kernel_size: tuple
    pool_size : tuple
    activation_func : Callable = nn.gelu
    use_batchnorm : bool = True

    @nn.compact
    def __call__(self, x, train):
        x = self.activation_func(x)
        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size,   
                    use_bias=False)(x)
        if self.use_batchnorm:     
            x = nn.BatchNorm(use_running_average = not train, 
                            momentum=0.90, 
                            epsilon=0.001)(x)  
        # jax.debug.print("{x}", x=x.shape)
        x = nn.max_pool(x, 
                        window_shape = self.pool_size, 
                        strides = self.pool_size, 
                        padding='SAME')
        return x


class conv_tower(nn.Module):
    kernel_size : tuple
    pool_size : tuple
    layer_features = np.array([288, 323, 363, 407, 456, 512])

    @nn.compact
    def __call__(self, x, train):
        for i in range(len(self.layer_features)):
            x = conv_block(features=self.layer_features[i],
                            kernel_size = self.kernel_size,
                            pool_size = self.pool_size
                          )(x, train)
        return x        


class dense_block(nn.Module):
    features : int
    dropout_rate : float = 0.2
    activation_func : Callable = nn.gelu
    use_batchnorm : bool = True

    @nn.compact
    def __call__(self, x, train):
        x = self.activation_func(x)
        x = jnp.reshape(x, (*x.shape[:-2], x.shape[-1]*x.shape[-2]))
        x = nn.Dense(features=self.features,
                    use_bias=False,
                    )(x) 
        if self.use_batchnorm: 
            x = nn.BatchNorm(use_running_average = not train, 
                            momentum=0.90, 
                            epsilon=0.001)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x


class Stc_rev_compl(nn.Module):
    rng_collection: str = 'stc_rev_compl'

    @nn.compact
    def __call__(self, x, deterministic: Optional[bool] = None):
        if deterministic:
            return x
        else:
            rng = self.make_rng(self.rng_collection)
            rev_cond = jax.random.uniform(rng)
            def rev(x):
                x = x[...,:,::-1]
                x = x[...,::-1,:]
                return x
        return jax.lax.cond(rev_cond>0.5, rev, lambda x: x, x)


class Stc_shift(nn.Module):
    rng_collection: str = 'stc_shift'
    shift_range : tuple = (-3, 3)
    pad_value : float = 0.25
    
    @nn.compact
    def __call__(self, x, deterministic : Optional[bool] = None) -> Any:
        if deterministic:
            return x
        else:
            rng = self.make_rng(self.rng_collection)
            if len(x.shape)>2: 
                rngs = jax.random.split(rng, x.shape[0])        #batch_dim
            else:
                rngs = rng
            def stc_shift_single(x, rng):
                padded_x = jnp.pad(x, ((-self.shift_range[0], self.shift_range[1]), (0,0)), 
                                   constant_values=self.pad_value)
                shift_pos = jax.random.choice(rng, jnp.arange(self.shift_range[0], self.shift_range[1]+1, dtype=int))
                start = shift_pos+(-self.shift_range[0])
                x = jax.lax.dynamic_slice(padded_x, (start, 0), x.shape)
                return x 
            
            x = jax.vmap(stc_shift_single, in_axes=(0, 0), out_axes=0)(x, rngs)
        return x
    

class bias(nn.Module):
    @nn.compact    
    def __call__(self, inputs):
        features = inputs.shape[-1]
        bias = self.param('bs', nn.initializers.zeros, (features,))
        out = inputs + bias
        return out


class MLP(nn.Module):
    features : tuple 
    activation : Callable = nn.softplus
    batchnormalize : bool = False
    use_bias : bool = True
    dropout_rate : float = 0.3

    @nn.compact
    def __call__(self, inputs, train):
        for i, ft in enumerate(self.features):
            inputs = nn.Dense(features=ft,
                              use_bias=self.use_bias,
                              )(inputs)
            if self.dropout_rate > 0:
                inputs = nn.Dropout(rate=self.dropout_rate)(inputs,
                                                            deterministic=not train)
            if i != len(self.features)-1:
                inputs = self.activation(inputs)
            if self.batchnormalize:
                inputs = nn.BatchNorm(use_running_average=not train,
                                      momentum=0.90, 
                                      epsilon=0.001)(inputs)    
        return inputs


class Decoder(nn.Module):
    n_cell : int 
    use_disp : bool = False
    use_bias : bool = True
    layers : int = 1
    activation : Callable = nn.softplus
    final_activation : bool = False
    
    @nn.compact
    def __call__(self, peakemb, cellemb):
        out = peakemb @ cellemb.T
        
        for i in np.arange(self.layers):
            if self.use_disp:
                disp = self.param(f'disp_{i}', nn.initializers.ones, (self.n_cell,))
                out *= disp
            if self.use_bias:    
                bias = self.param(f'bs_{i}', nn.initializers.zeros, (self.n_cell,))
                out += bias
            if (i != self.layers-1) or self.final_activation:
                out = self.activation(out)
            
        return out 
    


class GeneExpr2CellEmb(nn.Module):
    latent_dim : int = 32
    layer_features : tuple = (64,)          # high -> low
    
    @nn.compact
    def __call__(self, x):
        layer_features = (*self.layer_features, self.latent_dim)
        return MLP(features=layer_features)(x)

class CellEmb2GeneExpr(nn.Module):
    n_gene : int
    layer_features : tuple = (64,)           # high -> low
    
    @nn.compact
    def __call__(self, x):
        layer_features = (self.n_gene, *self.layer_features)
        return MLP(features=layer_features[::-1])(x)
    


def TF_conv(x: np.ndarray, filters: np.ndarray):
    dim_numbers = flax.linen.linear._conv_dimension_numbers(x.shape)
    filters = jnp.moveaxis(filters, 0, -1)
    out = jax.lax.conv_general_dilated(x, 
                                       filters, 
                                       window_strides=(1,), 
                                       dimension_numbers=dim_numbers,
                                       padding='same')
    return out      #(n_peak, 1344, n_filter)

class TF_conv_block(nn.Module):
    pool_size : tuple
    known_filter : bool = False
    infer_filter_num : int = 100
    kernel_size : int = 19
    activation_func : Callable = nn.gelu
    use_batchnorm : bool = True
    
    @nn.compact
    def __call__(self, seq, filters, train):
        if self.known_filter:
            x = TF_conv(seq, filters)
        
        if self.infer_filter_num > 0:
            y = self.activation_func(seq)
            y = nn.Conv(features=self.infer_filter_num, 
                        kernel_size=(self.kernel_size,),   
                        use_bias=False)(y) 
            if self.known_filter:
                x = jnp.concatenate((x, y), axis=2)
            else:
                x = y
        
        if self.use_batchnorm:
            x = nn.BatchNorm(use_running_average = not train, 
                            momentum=0.90, 
                            epsilon=0.001)(x)  
        x = nn.max_pool(x, 
                        window_shape = self.pool_size, 
                        strides = self.pool_size, 
                        padding='SAME')
        return x


class Seq2PeakEmb_orig(nn.Module):
    latent_dim : int
    known_filter : bool = True
    infer_filter_num : int = 0
    kernel_sizes : tuple = (17, 5, 5, 5, 5, 5, 5, 1)
    pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    activation_func : Callable = nn.gelu
    
    @nn.compact
    def __call__(self, x, filters, train):
        x = CNN(known_filter=self.known_filter,
                infer_filter_num=self.infer_filter_num,
                kernel_sizes=self.kernel_sizes,
                pool_sizes=self.pool_sizes,
                layer_features=self.layer_features,
                use_stc_rev_compl=False)(x, filters, train)
        
        x = dense_block(features=self.latent_dim)(x, train)
        x = self.activation_func(x)                                  #(n_peak, 32)        
        return x
    

class Seq2PeakEmb(nn.Module):
    latent_dim : int
    known_filter : bool = True
    infer_filter_num : int = 100
    kernel_sizes : tuple = (19,5,5)
    # pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    # layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    pool_sizes : tuple = (4,4,4)
    layer_features : tuple = (512, 256)
    activation_func : Callable = nn.gelu
    cnn_activation_func : Callable = nn.gelu
    dense_activation_func : Callable = nn.gelu
    
    
    @nn.compact
    def __call__(self, x, pwms, filters, train):
        cnn = CNN(known_filter=self.known_filter,
                infer_filter_num=self.infer_filter_num,
                kernel_sizes=self.kernel_sizes,
                pool_sizes=self.pool_sizes,
                layer_features=self.layer_features,
                activation_func=self.cnn_activation_func)
        
        x = cnn(x, filters, train)
        pwms = cnn.apply(cnn.variables, pwms, filters, False)
        pwms = self.dense_activation_func(pwms)
        
        # pwms = jnp.reshape(pwms, (*pwms.shape[:-2], pwms.shape[-1]*pwms.shape[-2]))
        pwms = jnp.max(pwms, axis=-1)
        
        x = dense_block(features=self.latent_dim,
                        activation_func=self.dense_activation_func)(x, train)
        x = self.activation_func(x)                                  #(n_peak, 32)        
        return x, pwms



class CNN(nn.Module):
    known_filter : bool = True
    infer_filter_num : int = 288
    kernel_sizes : tuple = (17, 5, 5, 5, 5, 5, 5, 1)
    pool_sizes : tuple = (3, 2, 2, 2, 2, 2, 2, 1)
    layer_features : tuple = (288, 323, 363, 407, 456, 512, 256)
    activation_func : Callable = nn.gelu
    use_stc_rev_compl : bool = True
    
    @nn.compact
    def __call__(self, x, filters, train):
        if self.use_stc_rev_compl:
            x = Stc_rev_compl()(x, deterministic= not train)
        x = Stc_shift()(x, deterministic= not train)
        x = TF_conv_block(pool_size=(self.pool_sizes[0],),
                          known_filter=self.known_filter,
                          infer_filter_num=self.infer_filter_num,
                          kernel_size=self.kernel_sizes[0],
                          activation_func=self.activation_func)(x, filters, train)     #(n_peak, 1344/4, n_filter)   
        
        for i,lf in enumerate(self.layer_features):
            x = conv_block(features = lf,
                            kernel_size=(self.kernel_sizes[i+1],),
                            pool_size=(self.pool_sizes[i+1],),
                            activation_func=self.activation_func)(x, train)
        return x
            






class attshare(nn.Module):
    use_att : bool = True
    qk_feature : tuple = 16
    tau : float = 40.
    @nn.compact
    def __call__(self, peakemb_acc, peakemb_rna, cellemb_rna_u, cellemb_rna_s, cellemb_acc, train):      #(peak, 32), (cell, 32)        
        if not self.use_att:
            return peakemb_rna, peakemb_rna, peakemb_acc
        
        q_mlp = nn.Dense(features=self.qk_feature,           
                        use_bias=True)
        q_acc = q_mlp(peakemb_acc)

        q_rna = q_mlp.apply(q_mlp.variables, peakemb_rna)          #(n_peak, 16)
        
        k_acc = nn.Dense(features=self.qk_feature,
                    use_bias=False)(cellemb_acc.T)          #(32dim, 16)
        k_rna_u = nn.Dense(features=self.qk_feature,
                    use_bias=False)(cellemb_rna_u.T)          #(32dim, 16)
        k_rna_s = nn.Dense(features=self.qk_feature,
                    use_bias=False)(cellemb_rna_s.T)

        c = GumbelSoftmaxAtt()(q_acc, k_acc, tau=self.tau, deterministic=not train)
        u = GumbelSoftmaxAtt()(q_rna, k_rna_u, tau=self.tau, deterministic=not train)
        s = GumbelSoftmaxAtt()(q_rna, k_rna_s, tau=self.tau, deterministic=not train)

        c, u, s = jnp.squeeze(c), jnp.squeeze(u), jnp.squeeze(s)                                          #(n_peak, 32)
        return u, s, c                                              #(n_peak, 32)

class GumbelSoftmaxAtt(nn.Module):
    rng_collection: str = 'gumbel'
    @nn.compact
    def __call__(self, q:np.ndarray, k:np.ndarray, tau:float=40., deterministic=False):
        qk_dim = q.shape[-1]
        a = q @ k.T/jnp.sqrt(qk_dim)            #(n_peak, 32)
        if not deterministic:
            rng = self.make_rng(self.rng_collection)              
            g = jax.random.gumbel(rng, shape=a.shape)
        else:
            g = 0.5772156649
        a = a+g
        a /= tau 
        a = jax.nn.softmax(a)
        return a 

class ModalDiscriminator(nn.Module):
    shared_cell_dim : int = 32
    @nn.compact
    def __call__(self, cellemb_acc, cellemb_rna, train):
        # cellemb_acc : (n_cell, 32)
        cellemb = jnp.concatenate((cellemb_acc[:,:self.shared_cell_dim], cellemb_rna[:,:self.shared_cell_dim]), axis=0)       #(2*n_cell, 32)
        cellemb = nn.BatchNorm(use_running_average = not train)(cellemb)
        cellemb = nn.Dense(features=1,           
                            use_bias=True)(cellemb)
        return cellemb                                    #(2*n_cell)             // sigmoid covered in loss 





