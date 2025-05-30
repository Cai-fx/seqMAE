import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial 
from sklearn.metrics import roc_auc_score, average_precision_score
from flax.linen import sigmoid
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import h5py


def single_rna_loss(y_pred, y_true, eps=1e-5):
    loss = optax.l2_loss(y_pred, y_true).mean()
    # loss = (y_pred - y_true*jnp.log(y_pred + eps)).mean()
    # loss = (1-optax.cosine_similarity(y_pred-jnp.mean(y_pred, axis=0), 
    #                                   y_true-jnp.mean(y_true, axis=0), epsilon=1e-5).mean()) + \
    #         (1-optax.cosine_similarity(y_pred-jnp.mean(y_pred, axis=1, keepdims=True), 
    #                                    y_true-jnp.mean(y_true, axis=1, keepdims=True), epsilon=1e-5).mean())
                            
    return loss


@partial(jax.jit, static_argnums=2)
def compute_metrics_rna(y_pred, labels, key='rna_expr'):

    def _corr(_y_pred, _label):
        return jnp.corrcoef(jnp.asarray((_y_pred, _label)))[0,1]
    corrs = jax.vmap(_corr, in_axes=(0,0), out_axes=0)(y_pred, labels)
    corrs = jnp.nan_to_num(corrs, 0.)
    corr = corrs.mean()
    # loss = optax.l2_loss(y_pred, labels).mean()
    loss = single_rna_loss(y_pred, labels)
    metrics = {
        f'{key}_loss' : loss,
        f'{key}_corrcoef': corr,
    }
    return metrics

@partial(jax.jit, static_argnums=2)
def compute_metrics_acc(y_pred, labels, key='acc'):
    loss = optax.sigmoid_binary_cross_entropy(y_pred, labels).mean()
    accuracy = jnp.mean((y_pred>=0.5) == labels)
    metrics = {
        f'{key}_loss': loss,
        f'{key}_accuracy': accuracy,
    }
    return metrics    
    
    
def auroc(y_true, y_pred):
    '''(n_peak, n_cell)'''
    auroc_per_gene = np.zeros(y_true.shape[0])
    auroc_per_cell = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[0]):
        try:
            auroc_per_gene[i] = roc_auc_score(y_true[i].flatten(), y_pred[i].flatten())
        except ValueError:
            auroc_per_gene[i] = np.nan
    for i in range(y_true.shape[1]):
        try:
            auroc_per_cell[i] = roc_auc_score(y_true[:,i].flatten(), y_pred[:,i].flatten())
        except ValueError:
            auroc_per_cell[i] = np.nan
        
    return {'auroc_per_cell': auroc_per_cell, 'auroc_per_peak': auroc_per_gene}


def auprc(y_true, y_pred):
    auprc_per_gene = np.zeros(y_true.shape[0])
    auprc_per_cell = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[0]):
        try:
            auprc_per_gene[i] = average_precision_score(y_true[i].flatten(), y_pred[i].flatten())
        except ValueError:
            auprc_per_gene[i] = np.nan
    for i in range(y_true.shape[1]):
        try:
            auprc_per_cell[i] = average_precision_score(y_true[:,i].flatten(), y_pred[:,i].flatten())
        except ValueError:
            auprc_per_cell[i] = np.nan
        
    return {'auprc_per_cell': auprc_per_cell, 'auprc_per_peak': auprc_per_gene}


def corrcoef(y_true, y_pred):
    '''(n_peak, n_cell)'''
    corrcoef_per_gene = np.zeros(y_true.shape[0])
    corrcoef_per_cell = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[0]):
        corrcoef_per_gene[i] = np.corrcoef([y_true[i].flatten(), y_pred[i].flatten()])[0,1]
    
    for i in range(y_true.shape[1]):
        corrcoef_per_cell[i] = np.corrcoef([y_true[:,i].flatten(), y_pred[:,i].flatten()])[0,1]
    
    return {'corr_per_cell': corrcoef_per_cell, 'corr_per_gene': corrcoef_per_gene}






def plot_cell_embeddings(ad_rna, Z=[], keys=['rna_u', 'rna_s','acc'], color_key="celltype"):
    fig, axes = plt.subplots(1,len(Z), figsize=(3*len(Z),3))
    if len(Z) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ad_rna.obsm[f'X_cell_emb_{keys[i]}'] = Z[i]      #(n_cell, 32)
        sc.pp.neighbors(ad_rna, use_rep=f'X_cell_emb_{keys[i]}')
        sc.tl.umap(ad_rna)
        if i != len(Z)-1:
            loc=None
        else:
            loc='right margin'
        sc.pl.umap(ad_rna, color=color_key, title=keys[i], ax=ax, show=False, legend_loc=loc)
        ad_rna.obsm[f'X_umap_{keys[i]}'] = ad_rna.obsm['X_umap']
    return fig


def plot_shared_embedding(ad_rna, Z=[], keys=[], color_key="celltype"):
    ad_comb = anndata.AnnData(np.concatenate(Z))
    keys_rep = [z.shape[0] for z in Z]
    ad_comb.obs['celltype'] =np.tile(ad_rna.obs[color_key], len(Z))
    ad_comb.obs['modaltype'] = np.asarray([[keys[i]]*keys_rep[i] for i in range(len(keys))]).flatten()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
    sc.pp.neighbors(ad_comb, use_rep='X')
    sc.tl.umap(ad_comb)
    for i, k in enumerate(keys):
        ad_rna.obsm[f'X_shared_umap_{k}'] = ad_comb.obsm['X_umap'][ad_rna.n_obs*i:ad_rna.n_obs*(i+1),:]

    ax1 = sc.pl.umap(ad_comb, color='celltype', ax=ax1, show=False)
    ax2 = sc.pl.umap(ad_comb, color='modaltype', ax=ax2, show=False)
    return fig






def _read_ids(pwm_model, ds_key="val", type="rna"):
    if type == 'rna':
        preprocess_folder = pwm_model.preprocess_folder_rna
        split_file = '%s/gene_splits.h5'%preprocess_folder
    if type == 'acc':
        preprocess_folder = pwm_model.preprocess_folder_acc
        split_file = '%s/splits.h5'%preprocess_folder

    with h5py.File(split_file, 'r') as hf:
        train_ids = hf[f'{ds_key}_ids'][:]
    return train_ids

def plot_dim_phases(Z0, Z1, Z0t=None, Z1t=None, xlabel="", ylabel="", colors=None):
    '''
    Z0: x
    Z1: y
    '''
    fig, axs = plt.subplots(4, 8, figsize=(5*8, 3*4), tight_layout=True)
    for i in range(32):
        axs[i//8, i%8].scatter(Z0[:,i], Z1[:,i], c=colors, s=1);
        
        _min, _max = np.min([Z0[:,i], Z1[:,i]]), \
                        np.max([Z0[:,i],Z1[:,i]])
        
        if Z0t is not None:
            axs[i//8, i%8].scatter(Z0t[:,i], Z1t[:,i], s=1);
            _min, _max = np.min([Z0[:,i], Z1[:,i], Z0t[:,i], Z1t[:,i]]), \
                        np.max([Z0[:,i],Z1[:,i], Z0t[:,i], Z1t[:,i]])
        axs[i//8, i%8].plot(np.linspace(_min, _max, 30), np.linspace(_min, _max, 30), alpha=0.5, color='grey');
    fig.supxlabel(xlabel);
    fig.supylabel(ylabel);
    return fig