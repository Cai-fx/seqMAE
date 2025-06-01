import jax 
from functools import partial
import flax.linen as nn
from .scb_blocks import MLP
import jax.numpy as jnp 
import flax 
import numpy as np 
import h5py 
from tqdm import tqdm 
from ..preprocessing.pwms import pad_pwm_df


def _grad_tf_func(model, ):
    def tf_2_acc(tf_expr, seq, atac_dpth):
        '''
        tf_expr: (),
        acc : (1344, 4)
        '''
        y = model.model.apply(
                    {"params": model.state.params,
                    "batch_stats": model.state.batch_stats},
                    seq,
                    tf_expr,
                    atac_dpth,
                    False,
                    False 
                    )
        y_acc = y[0]
        y_acc = jax.nn.sigmoid(y_acc).mean()
        return y_acc        
    
    grad_func = jax.grad(tf_2_acc, argnums=0)    
    grad_func = jax.jit(grad_func)
    return grad_func

@partial(jax.jit, static_argnames=["grad_func"])
def grad_tf(seq, tf_expr, atac_dpth, grad_func):            
    '''
    gradient for all tfs, evaluated at a sequence across all cells 
    tf_expr: (n_cell, n_tf,)  
    atac_dpth: (n_cell)
    ret: 
        (n_cell, n_tf)
    ''' 
    grads = jax.vmap(lambda tf_expr, atac_dpth: grad_func(tf_expr, seq, atac_dpth), in_axes=(0,0), out_axes=0)(
        tf_expr, atac_dpth
        )     #(n_cell, n_tf)
    return grads        #(n_cell, n_tf)


def write_grad_tf(model, all_ds, 
                  path = "./benchmark_summarize/gradtf_allcell.h5"):
    grad_func = _grad_tf_func(model=model)
    with h5py.File(path, 'w') as hf:
        data = hf.create_dataset('grads',
                                shape=(all_ds['x_acc'].shape[0], all_ds['rna'].shape[0], all_ds['rna'].shape[1]),
                                chunks=(5, all_ds['rna'].shape[0], 1),
                                dtype=np.float16)

        for i in tqdm(range(all_ds['x_acc'].shape[0])):
            seq = all_ds['x_acc'][i]
            grad = grad_tf(seq, tf_expr=all_ds['rna'], 
                            atac_dpth=np.repeat(np.mean(all_ds['atac_dpth'], keepdims=True), all_ds['atac_dpth'].shape[0]), 
                            grad_func=grad_func)

            grad = np.array(grad, dtype=np.float16)
            data.write_direct(grad, np.s_[:,:], np.s_[i,:,:])

    return



# @partial(jax.jit, static_argnames=["model"])
def _grad_seq_func(atac_dpth, model):
    def seq_2_acc(seq, tf_expr):
        '''
        tf_expr: (),
        seq : (... , 1344, 4)
        '''
        res = model.model.apply(
                    {"params": model.state.params,
                    "batch_stats": model.state.batch_stats},
                    seq,                                #(1344, 4)
                    tf_expr,                            #(n_tf, )
                    atac_dpth,
                    False,
                    False 
                    )                                   #(1, n_cell)
        y_acc = res[0]
        y_acc = jax.nn.sigmoid(y_acc).mean()            #(1, n_cell) -> (1)
        return y_acc        
    grad_func = jax.grad(seq_2_acc, argnums=0)    
    grad_func = jax.jit(grad_func)
    return grad_func

@partial(jax.jit, static_argnames=["grad_func"]) 
def grad_func_batched(seq, tf_expr, grad_func):
    grads = jax.vmap(lambda c: grad_func(seq, c), 
                                        in_axes=(0,), out_axes=0)(
                                            tf_expr
                                            )   
    return grads

def grad_seq(seq, tf_expr, grad_func, batch_size=1000):
    '''
    gradient wrt a sequence, mean across some cells 
    tf_expr: (n_cell, n_tf)        ->  input cell 
    batch_size: over cells 
    ''' 
    
    grads = np.zeros((tf_expr.shape[0], *seq.shape)) 
    for i_start in range(0, tf_expr.shape[0], batch_size):
        i_end = min(i_start+batch_size, grads.shape[0])
        grads[i_start:i_end] = grad_func_batched(seq=seq, 
                                                 tf_expr=tf_expr[i_start:i_end],
                                                 grad_func=grad_func
                                                 )     ## (n_cell, 1344, 4)
    ## normalize gradient?
    # grads = grads/jnp.sum(jnp.abs(grads))
    return grads     

def write_grad_seq(model, 
                   jaspar_motifs,
                   all_ds,
                   path="./benchmark_summarize/gradSeq_allcell.h5", 
                   batch_size = 500,
                   batch_size_cell = 100,
                   ):
    pwms = pad_pwm_df(jaspar_motifs)           
    pwms_combined = np.concatenate([pwms, pwms[:,::-1,::-1]], axis=0)         ## (n_filt*2, filt_len, d)
    norm_filters = jnp.sum(jnp.square(pwms_combined), axis=(1,2))
    norm_filters = jnp.sqrt(norm_filters)                               #(n_filt,)
    
    with h5py.File(path, 'w') as hf:
        data = hf.create_dataset('grads',
                                shape=(all_ds['x_acc'].shape[0], all_ds['rna'].shape[0], len(jaspar_motifs)),
                                chunks=(5, all_ds['rna'].shape[0], 1),
                                dtype=np.float16) 
        
        atac_dpth = np.mean(all_ds['atac_dpth'], keepdims=True)
        grad_func = _grad_seq_func(atac_dpth, model)
        
        for i in tqdm(range(all_ds['x_acc'].shape[0])):
            seq = all_ds['x_acc'][i]
            grad = grad_seq(seq, tf_expr=all_ds['rna'], 
                            grad_func=grad_func,
                            batch_size=batch_size
                            )       #(n_cell, 1344, 4)

            conv_seq = np.zeros((all_ds['rna'].shape[0], len(jaspar_motifs)), dtype=np.float16)   #(n_cell, n_motif)
            for i_start in range(0, grad.shape[0], batch_size_cell):
                i_end = min(i_start+batch_size_cell, grad.shape[0])
                conv_seq[i_start:i_end] = score_conv(grad_x_input=grad[i_start:i_end, :, :]*seq,
                                                        pwms = pwms_combined,
                                                        n_filt = pwms.shape[0],
                                                        norm_filters=norm_filters,
                                                        )                   #(n_cell, n_motif)
                
            # data[i] = np.array(conv_seq)                                    #(n_peak, n_cell, n_motif)
            data.write_direct(np.array(conv_seq, dtype=np.float16), np.s_[:,:], np.s_[i,:,:])

    return



def get_grad_tf_small(model, all_ds, ):
    grad_func = _grad_tf_func(model=model)
    grads = np.zeros((all_ds['x_acc'].shape[0], all_ds['rna'].shape[0], all_ds['rna'].shape[1]))

    for i in tqdm(range(all_ds['x_acc'].shape[0])):
        seq = all_ds['x_acc'][i]
        grad = grad_tf(seq, tf_expr=all_ds['rna'], 
                        atac_dpth=np.repeat(np.mean(all_ds['atac_dpth'], keepdims=True), all_ds['atac_dpth'].shape[0]), 
                        grad_func=grad_func)
        
        grads[i] = grad 

    return grads 

def get_grad_seq_small(model, 
                   jaspar_motifs,
                   all_ds,
                   batch_size = 500,
                   batch_size_cell = 100,
                   ):
    pwms = pad_pwm_df(jaspar_motifs)           
    pwms_combined = np.concatenate([pwms, pwms[:,::-1,::-1]], axis=0)         ## (n_filt*2, filt_len, d)
    norm_filters = jnp.sum(jnp.square(pwms_combined), axis=(1,2))
    norm_filters = jnp.sqrt(norm_filters)                               #(n_filt,)
    
    grads = np.zeros((all_ds['x_acc'].shape[0], all_ds['rna'].shape[0], len(jaspar_motifs)))
    
    atac_dpth = np.mean(all_ds['atac_dpth'], keepdims=True)
    grad_func = _grad_seq_func(atac_dpth, model)
    
    for i in tqdm(range(all_ds['x_acc'].shape[0])):
        seq = all_ds['x_acc'][i]
        grad = grad_seq(seq, tf_expr=all_ds['rna'], 
                        grad_func=grad_func,
                        batch_size=batch_size
                        )       #(n_cell, 1344, 4)

        conv_seq = np.zeros((all_ds['rna'].shape[0], len(jaspar_motifs)), dtype=np.float16)   #(n_cell, n_motif)
        for i_start in range(0, grad.shape[0], batch_size_cell):
            i_end = min(i_start+batch_size_cell, grad.shape[0])
            conv_seq[i_start:i_end] = score_conv(grad_x_input=grad[i_start:i_end, :, :]*seq,
                                                    pwms = pwms_combined,
                                                    n_filt = pwms.shape[0],
                                                    norm_filters=norm_filters,
                                                    )                   #(n_cell, n_motif)
            
        # data[i] = np.array(conv_seq)                                    #(n_peak, n_cell, n_motif)
        grads[i, i_start:i_end, :] = conv_seq 
    return grads






class helper_model(nn.Module):
    encoder_features: tuple = (32, )
    mlp_activation = nn.gelu 
    mlp_batchnorm:bool = True 
    @nn.compact 
    def __call__(self, rna, Z_seq, train):
        rna = MLP(features=self.encoder_features,
                                activation=self.mlp_activation,
                                batchnormalize=self.mlp_batchnorm)(rna, train)               #(n_cell, d)
        seq = Z_seq @ rna.T                                                                         #(n_peak=1, n_cell=1)
        return seq 

def gradient_Z_seq(Z_seq, model, rna):
    '''
    dy_atac/d_Z
    Z_seq: (n_peak, 32) 
    rna: (n_cell, n_tf)
    '''
    
    def apply_2_acc(seq_Z, tf_expr):
        y_acc = helper_model(encoder_features = model.model_config['encoder_features'],
                             mlp_batchnorm = model.model_config['mlp_batchnorm']
                             ).apply({"params": model.state.params,
                                    "batch_stats": model.state.batch_stats},
                                   tf_expr,
                                   seq_Z,
                                   False)
        y_acc = jax.nn.sigmoid(y_acc).mean()
        return y_acc 
    
    grad_func = jax.grad(apply_2_acc, argnums=0)
    grad_func = jax.jit(grad_func)
    
    grads_f = jax.vmap(grad_func, in_axes=(None, 0), out_axes=0)
    grads = jax.vmap(lambda Z_seq, rna: grads_f(Z_seq, rna), in_axes=(0, None), out_axes=0)(Z_seq, rna)
    return grads



### compute convolution score for gradSeq 
def norm_conv(x, filters, norm_filters):
    '''
    x: (n_sample, len, d)
    filter: (2*n_filt, filt_l, d)  
    '''
    dim_numbers = flax.linen.linear._conv_dimension_numbers(x.shape)
    filters = jnp.moveaxis(filters, 0, -1)       
    out = jax.lax.conv_general_dilated(x, 
                                        filters, 
                                        window_strides=(1,), 
                                        dimension_numbers=dim_numbers,
                                        padding='Valid')                #(n_sample, len, n_filt*2)
    ## other score? 
    return out/(norm_filters)                                           #(n_sample, len, n_filt, )

@partial(jax.jit, static_argnames=["n_filt"])
def score_conv(grad_x_input:np.ndarray, pwms:np.ndarray, n_filt:int, norm_filters:np.ndarray):
    '''
    grad_x_input : (n_sample, len, 4)
    pwms: (n_filts, filt_len, 4) 
    '''
    conv_res = norm_conv(grad_x_input,
                         filters=pwms,
                         norm_filters=norm_filters)       ## also score reverse compliment -> (n_sample, 1344, 2*n_filt)
    conv_res = jnp.max(jnp.abs(conv_res), axis=1)      ## (n_sample, 2*n_filt)
    orig_scores = conv_res[:, :n_filt]
    rc_scores = conv_res[:, n_filt:]
    
    # Take maximum between original and RC scores
    return jnp.maximum(orig_scores, rc_scores)  # (n_sample, n_filt)