import numpy as np 
import pandas as pd 
import h5py 
from tqdm import tqdm 
import anndata 
from sklearn.linear_model import LogisticRegression
from typing import List, Optional
from sklearn.metrics import average_precision_score


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


def read_grads_celltype(tfs_cts,
                        jaspar_motifs,
                        ad_atac,
                        ad_rna,
                        _chunk_size = 50,
                        
                        ):
    """
    read gradTF & gradSeq by tf_ct pairs, averaged over some cells 
    (n_peak, n_cell, n_motif=1) -> (n_peak, n_motif=1)
    Ret  
    """
    tf_mm_ct = [] 
    for tf_ct in tfs_cts:
        tf, ct = tf_ct.split("_")[0], tf_ct.split("_")[1]
        mm = jaspar_motifs[jaspar_motifs['tf']==tf]['motif']
        tf_mm_ct.extend([tf+":"+m+"_"+ct for m in mm])

    gradTF = pd.DataFrame(index=ad_atac.var.index, columns=tfs_cts)
    gradSeq = pd.DataFrame(index=ad_atac.var.index, columns=tf_mm_ct)

    for i, m_idx in enumerate(jaspar_motifs.index):
        tf_name = jaspar_motifs.loc[m_idx]['tf']
        tf_idx = np.where(ad_rna.var['gene_symbols']==tf_name)[0][0]
        motif_name = jaspar_motifs.loc[m_idx]['motif']
        print(tf_name, motif_name)
        
        gradSeq_all, gradTF_all = read_grads_allcells(
            gradSeq_file="./benchmark_summarize/gradSeq_allcell.h5", 
            gradTF_file="./benchmark_summarize/gradtf_allcell.h5", 
            motif_idx=m_idx, 
            tf_idx=tf_idx,
            jaspar_motifs=jaspar_motifs,
            _chunk_size=_chunk_size
        )                                               #(peak, cell, motif=1)
        
        cts = tfs_cts.str.split("_").str[1][tfs_cts.str.split("_").str[0]==tf_name]
        for ct in cts:
            ct_idx = np.where(ad_rna.obs['celltype']==ct)[0]
            print(f"{ct}: {len(ct_idx)}")
            gradTF.loc[:,f"{tf_name}_{ct}"] = np.mean(gradTF_all[:,ct_idx], axis=1)        #(n_peak, n_cell)
            gradSeq.loc[:,f"{tf_name}:{motif_name}_{ct}"] = np.mean(gradSeq_all[:,ct_idx], axis=1)

    return gradSeq, gradTF 

# def regression_allcells(gradSeq_compiled, gradTF_compiled, 
#                         fimo_pval,
#                         chip_tissue,
#                         random_seed = 9876, down_sample_frac=0.3,
#                         plot_knee = True,
#                         select_hvp = True,
#                         ):
#     np.random.seed(random_seed)
#     n_cell_sample = int(gradSeq_compiled.shape[1]*down_sample_frac)
#     down_sample_cell_ids = np.random.randint(gradSeq_compiled.shape[1] ,size=(n_cell_sample,))
#     # down_sample_cell_ids.shape
    
#     X = np.stack((
#         gradTF_compiled[:, down_sample_cell_ids].flatten(), 
#         gradSeq_compiled[:, down_sample_cell_ids].flatten(),
#         np.column_stack([fimo_pval]*n_cell_sample).flatten()
#         )).T
#     y = np.column_stack([np.array(chip_tissue)]*n_cell_sample) 
#     logReg = LogisticRegression(penalty='none')
#     logReg.fit(X, y=y.flatten())

#     y_pred = np.zeros_like(gradSeq_compiled)
#     for c in tqdm(np.arange(gradSeq_compiled.shape[1])):
#         XX = np.stack((gradTF_compiled[:, c], 
#                     gradSeq_compiled[:, c],
#                     fimo_pval)).T
#         yy = logReg.predict_proba(XX)[:,1]
#         y_pred[:,c] = yy

#     # yy = np.sum(y_pred>0.5, axis=1)/y_pred.shape[1]         ## prop of cells with positive peaks in all cells 
#     # yy = (y_pred - y_pred.mean(axis=1, keepdims=True))/np.std(y_pred, axis=1, keepdims=True)
    
#     if select_hvp:
#         yy = np.var(y_pred, axis=1)         # var cross cells 
#         y_sort = yy[np.argsort(yy)]

#         kl = KneeLocator(np.arange(y_sort.shape[0]), y_sort, 
#                         curve="convex",
#                         direction="increasing", 
#                         interp_method="polynomial", 
#                         #  polynomial_degree=2,
#                         online=True)     
#         cutoff = y_sort[kl.elbow]
#         hv_peaks = np.where(yy>cutoff)[0] 
#         if plot_knee:     
#             kl.plot_knee()
        
#         tf_act_crosspeak = np.mean(y_pred[hv_peaks, :], axis=0)     #(n_cell,)
#     else:
#         tf_act_crosspeak = np.mean(y_pred, axis=0)
#     return tf_act_crosspeak




# def tf_activity_allcells(gradSeq_file="./benchmark_summarize/gradSeq_allcell.h5", 
#                          gradTF_file="./benchmark_summarize/gradtf_allcell.h5", 
#                          motif_idx = 67, 
#                          jaspar_motifs=pd.DataFrame([]),
#                          fimo_pvals = pd.DataFrame([]),
#                          random_seed = 9876,
#                          down_sample_frac = 0.3,
#                          ad_rna = None,
#                          chip_bulk = None,
#                          temp_path=None,
#                          _chunk_size=1000,
#                          select_hvp=True):
#     """logistic regression with 4 metrics"""
#     tf_name = jaspar_motifs.loc[motif_idx]['tf']
#     tf_idx = np.where(ad_rna.var['gene_symbols']==tf_name)[0][0]
#     motif_name = jaspar_motifs.loc[motif_idx]['motif']
#     print(tf_name, motif_name)
#     grad_seq, grad_tf = read_grads_allcells(gradSeq_file=gradSeq_file, 
#                                             gradTF_file=gradTF_file, 
#                                             motif_idx = motif_idx, 
#                                             tf_idx = tf_idx,
#                                             jaspar_motifs=jaspar_motifs,
#                                             temp_path=temp_path,
#                                             _chunk_size=_chunk_size)
    
#     chip_tissue = chip_bulk.loc[:, tf_name]
#     fimo_pval = fimo_pvals.loc[:,f"{tf_name}:{motif_name}"].to_numpy()
#     tf_act_crosspeak = regression_allcells(gradSeq_compiled=grad_seq, gradTF_compiled=grad_tf, 
#                                            fimo_pval = fimo_pval,
#                                         chip_tissue=chip_tissue,
#                                         random_seed = random_seed, down_sample_frac=down_sample_frac,
#                                         plot_knee = True,
#                                         select_hvp = select_hvp
#                                         )
    
#     return tf_act_crosspeak



class tf_act_logReg:
    def __init__(self, 
                 X, 
                 Y, 
                 Y_test,
                 downsample_frac=1.,
                 label=""):
        self.X = X              ## (n_peak, n_feature)
        self.Y = Y 
        self.Y_test = Y_test
        self.label = label 
        self.split_traintest(downsample_frac=downsample_frac)

        self.logReg = LogisticRegression(penalty=None, max_iter=100000)
        self.logReg.fit(self.X[self.train_id, ], y=self.Y[self.train_id])        
        self.y_pred = self.logReg.predict_proba(X)[:,1]
        self.eval_metrics()
        return 
    
    def split_traintest(self, 
                        downsample_frac = 0.01,
                        seed=4444,
                      ):
        n_peaks = self.X.shape[0] 
        if downsample_frac==1.:
            self.train_id = np.arange(n_peaks)
            return
        n_train = int(downsample_frac*n_peaks)
        np.random.seed(seed)
        train_id = np.random.choice(np.arange(n_peaks), size=n_train, replace=False)
        train_id.sort()
        self.train_id = train_id 
        return
    
    def eval_metrics(self,):
        self.auprc = average_precision_score(y_score=self.y_pred, y_true=self.Y_test)
        return
    
    


class tf_act_cross_peaks:
    def __init__(self, 
                 gt:pd.DataFrame,
                 metrics:List[pd.DataFrame],
                 jaspar_motifs:pd.DataFrame,     ## for referencing motifs 
                 chip_bulk:Optional[pd.DataFrame] = None,
                 ):
        self.gt = gt
        self.metrics=metrics 
        self.chip_bulk = chip_bulk
        self.jaspar_motifs = jaspar_motifs
        
        peak_shared = self.gt.index 
        for m in self.metrics:
            peak_shared = peak_shared.intersection(m.index)
        
        if not self.chip_bulk is None:
            peak_shared = peak_shared.intersection(self.chip_bulk.index) 
            self.chip_bulk = chip_bulk.loc[peak_shared,:]
        
        self.gt = self.gt.loc[peak_shared, :]
        for i, df in enumerate(metrics):
            self.metrics[i] = df.loc[peak_shared, :]
        pass
    
    def _get_metric_label_type(self, metric):
        if len(metric.columns.str.split("_")[0])>1:
            _tf_format = metric.columns[0].split("_")[0].split(":")
            label_type = "tf:mm_ct" if len(_tf_format) >1 else "tf_ct"
        else:
            if len(metric.columns[0].split(":"))>1:
                label_type = "tf:mm"  
            elif metric.columns[0] in self.jaspar_motifs['tf']:
                label_type = "tf"
            else:
                label_type="other"
                
        return label_type
    
    
    @staticmethod
    def _get_tfmmct(metric:pd.DataFrame, label_type:str):
        tf, mm, ct = None, None, None
        if "tf" in label_type:
            tf_mm = metric.columns.str.split("_").str[0]
            if "tf:mm" in label_type:
                tf, mm = tf_mm.str.split(":").str[0], tf_mm.str.split(":").str[1]
            else:
                tf = tf_mm 
        if "ct" in label_type:
            ct =  metric.columns.str.split("_").str[1] 
        return tf, mm, ct  
    
    def tidy_metrics(self, tf_gt=None, ct_gt=None, mm_gt=None):
        """
        tf_gt, ct_gt: label to match 
        """
        for i, metric in enumerate(self.metrics):
            label_type = self._get_metric_label_type(metric=metric)
            ## match tf & ct if both fields are present 
            m_sub = metric
            tf, mm, ct = tf_act_cross_peaks._get_tfmmct(metric=m_sub, label_type=label_type)
            if "tf" in label_type and not (tf_gt is None):
                m_sub = m_sub.loc[:,tf==tf_gt] 
                tf, mm, ct = tf_act_cross_peaks._get_tfmmct(metric=m_sub, label_type=label_type)
            if "ct" in label_type and not (ct_gt is None):
                m_sub = m_sub.loc[:,ct==ct_gt]
                tf, mm, ct = tf_act_cross_peaks._get_tfmmct(metric=m_sub, label_type=label_type)
            if "mm" in label_type and not (mm_gt) is None:
                m_sub = m_sub.loc[:, mm==mm_gt]
            
            _m_np = m_sub.to_numpy()
            X = _m_np if i==0 else np.concatenate((X, _m_np), axis=1)
            #(n_peak, n_feature)
        return X 
    
    
    def logReg_all(self, downsample_frac=1., trainOn="chip_bulk"):
        self.logRegs_all = {}
        for tf_ct in tqdm(self.gt.columns): 
            tf, ct = tf_ct.split("_")[0], tf_ct.split("_")[1]
            motifs = self.jaspar_motifs[self.jaspar_motifs['tf']==tf]['motif']
            for mm in motifs:
                X = self.tidy_metrics(tf_gt=tf, ct_gt=ct, mm_gt=mm)
                if trainOn=="chip_bulk":
                    Y = self.chip_bulk.loc[:,tf].to_numpy()
                elif trainOn=="gt":
                    Y = self.gt.loc[:,tf_ct].to_numpy()
                else:
                    raise Exception(f"trainOn = 'chip_bulk' or 'gt'")
                                
                logreg = tf_act_logReg(X=X, 
                                       Y=Y, 
                                       Y_test=self.gt.loc[:,tf_ct].to_numpy(),
                                       downsample_frac=downsample_frac,
                                       label=f"{tf}:{mm}_{ct}")
                
                self.logRegs_all[f"{tf}:{mm}_{ct}"] = logreg 
        return
    
    def auprcs_to_csv(self,):
        d_out = {}
        for k, v in self.logRegs_all.items():
            d_out[k] = [v.auprc]
        return pd.DataFrame.from_dict(d_out)
    
    
    
    
    
    
    