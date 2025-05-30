import numpy as np 
import pandas as pd 
import h5py 
from tqdm import tqdm 
import anndata 
from sklearn.linear_model import LogisticRegression
from kneed import KneeLocator

def read_grads_allcells(gradSeq_file="./benchmark_summarize/gradSeq_allcell.h5", 
                         gradTF_file="./benchmark_summarize/gradtf_allcell.h5", 
                         motif_idx = 67, 
                         tf_idx = 150,
                         jaspar_motifs=pd.DataFrame([]),
                         temp_path=None,
                         _chunk_size=1,):
    tf_name = jaspar_motifs.loc[motif_idx]['tf']
    motif_name = jaspar_motifs.loc[motif_idx]['motif']
    
    ## read gradSeq 
    with h5py.File(gradSeq_file, "r") as hf:
        grad_seq = hf[f'grads']                         #(n_peak, n_cell, n_motif)
        shape = grad_seq.shape
        chunk_size = (_chunk_size, shape[1], shape[2]) 
        gradSeq_all = np.zeros((shape[0], shape[1]))
        for i in tqdm(range(0, shape[0], chunk_size[0])):
            end_idx = min(i + chunk_size[0], shape[0])
            # chunk = grad_seq[i:end_idx,:,motif_idx]
            # 
            grad_seq.read_direct(gradSeq_all, np.s_[i:end_idx,:,motif_idx], np.s_[i:end_idx,:])
                
    ## read gradTF                                   #(n_peak, n_cell, n_tf)
    with h5py.File(gradTF_file, "r") as hf:
        grad_tf = hf[f'grads']
        shape = grad_tf.shape
        chunk_size = (_chunk_size, shape[1], shape[2]) 
        gradTF_all = np.zeros((shape[0], shape[1]))
        for i in tqdm(range(0, shape[0], chunk_size[0])):
            end_idx = min(i + chunk_size[0], shape[0])
            # chunk = grad_tf[i:end_idx,:,tf_idx]
            # mean_array_2[i:end_idx, :] = chunk
            grad_tf.read_direct(gradTF_all, np.s_[i:end_idx,:,tf_idx], np.s_[i:end_idx,:])
    # grad_tf = mean_array_2                              #(n_peak, n_cell)
    
    if not (temp_path is None):
        np.save(f"{temp_path}/gradTF_noabs_{tf_name}_allcells.npy", grad_tf)
        np.save(f"{temp_path}/gradSeq_{tf_name}_{motif_name}_allcells.npy", grad_seq)
    
    return gradSeq_all, gradTF_all

def regression_allcells(gradSeq_compiled, gradTF_compiled, 
                        fimo_pval,
                        chip_tissue,
                        random_seed = 9876, down_sample_frac=0.3,
                        plot_knee = True,
                        select_hvp = True,
                        ):
    np.random.seed(random_seed)
    n_cell_sample = int(gradSeq_compiled.shape[1]*down_sample_frac)
    down_sample_cell_ids = np.random.randint(gradSeq_compiled.shape[1] ,size=(n_cell_sample,))
    # down_sample_cell_ids.shape
    
    X = np.stack((
        gradTF_compiled[:, down_sample_cell_ids].flatten(), 
        gradSeq_compiled[:, down_sample_cell_ids].flatten(),
        np.column_stack([fimo_pval]*n_cell_sample).flatten()
        )).T
    y = np.column_stack([np.array(chip_tissue)]*n_cell_sample) 
    logReg = LogisticRegression(penalty='none')
    logReg.fit(X, y=y.flatten())

    y_pred = np.zeros_like(gradSeq_compiled)
    for c in tqdm(np.arange(gradSeq_compiled.shape[1])):
        XX = np.stack((gradTF_compiled[:, c], 
                    gradSeq_compiled[:, c],
                    fimo_pval)).T
        yy = logReg.predict_proba(XX)[:,1]
        y_pred[:,c] = yy

    # yy = np.sum(y_pred>0.5, axis=1)/y_pred.shape[1]         ## prop of cells with positive peaks in all cells 
    # yy = (y_pred - y_pred.mean(axis=1, keepdims=True))/np.std(y_pred, axis=1, keepdims=True)
    yy = np.var(y_pred, axis=1)         # var cross cells 
    y_sort = yy[np.argsort(yy)]

    kl = KneeLocator(np.arange(y_sort.shape[0]), y_sort, 
                    curve="convex",
                    direction="increasing", 
                    interp_method="polynomial", 
                    #  polynomial_degree=2,
                    online=True)     
    cutoff = y_sort[kl.elbow]
    hv_peaks = np.where(yy>cutoff)[0] 
    if plot_knee:     
        kl.plot_knee()
    
    if select_hvp:
        tf_act_crosspeak = np.mean(y_pred[hv_peaks, :], axis=0)     #(n_cell,)
    else:
        tf_act_crosspeak = np.mean(y_pred, axis=0)
    return tf_act_crosspeak

def tf_activity_allcells(gradSeq_file="./benchmark_summarize/gradSeq_allcell.h5", 
                         gradTF_file="./benchmark_summarize/gradtf_allcell.h5", 
                         motif_idx = 67, 
                         jaspar_motifs=pd.DataFrame([]),
                         fimo_pvals = pd.DataFrame([]),
                         random_seed = 9876,
                         down_sample_frac = 0.3,
                         ad_rna = None,
                         chip_bulk = None,
                         temp_path=None,
                         _chunk_size=1000,
                         select_hvp=True):
    tf_name = jaspar_motifs.loc[motif_idx]['tf']
    tf_idx = np.where(ad_rna.var['gene_symbols']==tf_name)[0][0]
    motif_name = jaspar_motifs.loc[motif_idx]['motif']
    print(tf_name, motif_name)
    grad_seq, grad_tf = read_grads_allcells(gradSeq_file=gradSeq_file, 
                                            gradTF_file=gradTF_file, 
                                            motif_idx = motif_idx, 
                                            tf_idx = tf_idx,
                                            jaspar_motifs=jaspar_motifs,
                                            temp_path=temp_path,
                                            _chunk_size=_chunk_size)
    
    chip_tissue = chip_bulk.loc[:, tf_name]
    fimo_pval = fimo_pvals.loc[:,f"{tf_name}:{motif_name}"].to_numpy()
    tf_act_crosspeak = regression_allcells(gradSeq_compiled=grad_seq, gradTF_compiled=grad_tf, 
                                           fimo_pval = fimo_pval,
                                        chip_tissue=chip_tissue,
                                        random_seed = random_seed, down_sample_frac=down_sample_frac,
                                        plot_knee = True,
                                        select_hvp = select_hvp
                                        )
    
    return tf_act_crosspeak