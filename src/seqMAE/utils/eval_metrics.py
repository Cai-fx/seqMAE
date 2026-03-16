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
