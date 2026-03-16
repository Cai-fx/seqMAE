import numpy as np 
import pandas as pd 
import h5py 
from tqdm import tqdm 
import anndata 
from sklearn.linear_model import LogisticRegression
from typing import List, Optional
from sklearn.metrics import average_precision_score
import jax 
from functools import partial
import flax.linen as nn
from .scb_blocks import MLP
import jax.numpy as jnp 
import flax 
from ..preprocessing.other_pp import pad_pwm_df


### ------------------ compute gradients ------------------
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
    """gradient for all tfs, evaluated at a sequence across all cells

    Args:
        seq (_type_): _description_
        tf_expr (_type_): (n_cell, n_tf,)
        atac_dpth (_type_): (n_cell)
        grad_func (_type_): _description_

    Returns:
        _type_: (n_cell, n_tf)
    """
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




### ------------------ read/write gradients ------------------
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





### ------------------ logistic regressions ------------------
#### alternative wayas to combine the metrics
import numpy as np 
import pandas as pd 
import h5py 
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from typing import List, Optional
from sklearn.metrics import average_precision_score, f1_score
from scipy.special import logit 


class tf_act_Reg:
    def __init__(self, 
                 X, 
                 Y, 
                 Y_test,
                 downsample_frac=1.,
                 label="",
                 mode="logReg",
                 **model_kwargs):
        self.X = X                      ## (n_peak, n_feature)
        self.Y = Y 
        self.Y_test = Y_test
        self.label = label 
        self.split_traintest(downsample_frac=downsample_frac)
        
        if (mode=="logReg"):
            self.model = LogisticRegression(penalty=None, max_iter=100000)    
        elif (mode == "randomForest"):
            self.model = RandomForestClassifier(**model_kwargs)
        elif (mode=="XGBoost"):
            self.model = GradientBoostingClassifier(**model_kwargs)
        elif (mode =="MLP"):
            self.model = MLPClassifier(**model_kwargs)
        elif (mode == "svc"):
            self.model = SVC(**model_kwargs)
        else:
            raise Exception("mode error")
        
        self.model.fit(self.X[self.train_id, ], y=self.Y[self.train_id])  
        
        if (mode=="logReg" or mode=="randomForest" or mode=="XGBoost" or mode=="MLP") : 
            self.y_pred = self.model.predict_proba(X)[:,1]
        elif (mode == "svc"):
            self.y_pred = self.model.predict(X)
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
        self.f1 = f1_score(y_true=self.Y_test, y_pred=(self.y_pred>0.5))
        return

class tf_act_cross_peaks_base:
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
        
        self.peak_shared = peak_shared
        pass
    
    def _get_metric_label_type(self, metric):
        if len(metric.columns.str.split("_")[0])>1:
            _tf_format = metric.columns[0].split("_")[0].split(":")
            label_type = "tf:mm_ct" if len(_tf_format) >1 else "tf_ct"
        else:
            if len(metric.columns[0].split(":"))>1:
                label_type = "tf:mm"  
            elif metric.columns[0] in self.jaspar_motifs['tf'].to_list():
                label_type = "tf"
            elif len([cc for cc in metric.columns if cc in self.gt.columns.str.split("_").str[1]])>0:
                label_type = "ct"
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
            if (label_type == "tf:mm_ct") or (label_type == "tf_ct"):
                ct =  metric.columns.str.split("_").str[1] 
            elif (label_type=="ct"):
                ct = metric.columns
        return tf, mm, ct  
    
    def tidy_metrics(self, tf_gt=None, ct_gt=None, mm_gt=None):
        """
        tf_gt, ct_gt: label to match 
        """
        for i, metric in enumerate(self.metrics):
            label_type = self._get_metric_label_type(metric=metric)
            ## match tf & ct if both fields are present 
            m_sub = metric
            tf, mm, ct = tf_act_cross_peaks_alt._get_tfmmct(metric=m_sub, label_type=label_type)
            if "tf" in label_type and not (tf_gt is None):
                m_sub = m_sub.loc[:,tf==tf_gt] 
                tf, mm, ct = tf_act_cross_peaks_alt._get_tfmmct(metric=m_sub, label_type=label_type)
            if "ct" in label_type and not (ct_gt is None):
                m_sub = m_sub.loc[:,ct==ct_gt]
                tf, mm, ct = tf_act_cross_peaks_alt._get_tfmmct(metric=m_sub, label_type=label_type)
            if "mm" in label_type and not (mm_gt) is None:
                m_sub = m_sub.loc[:, mm==mm_gt]
            
            _m_np = m_sub.to_numpy()
            X = _m_np if i==0 else np.concatenate((X, _m_np), axis=1)
            #(n_peak, n_feature)
        return X 
    

class tf_act_cross_peaks_alt(tf_act_cross_peaks_base):
    def __init__(self, gt, metrics, jaspar_motifs, chip_bulk = None):
        super().__init__(gt, metrics, jaspar_motifs, chip_bulk)
        pass
    
    def logReg_all(self, downsample_frac=1., trainOn="chip_bulk", mode="logReg", **model_kwargs):
        self.logRegs_all = {}
        y_pred = {}
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
                                
                logreg = tf_act_Reg(X=X, 
                                    Y=Y, 
                                    Y_test=self.gt.loc[:,tf_ct].to_numpy(),
                                    downsample_frac=downsample_frac,
                                    label=f"{tf}:{mm}_{ct}", 
                                    mode=mode,
                                    **model_kwargs)
                
                self.logRegs_all[f"{tf}:{mm}_{ct}"] = logreg 
                y_pred[f"{tf}:{mm}_{ct}"] = logreg.y_pred
        self.y_pred = pd.DataFrame(y_pred, index=self.gt.index)
        return
    
    def auprcs_to_csv(self, metric="auprc"):
        d_out = {}
        for k, v in self.logRegs_all.items():
            if metric == "auprc":
                d_out[k] = [v.auprc]
            if metric == "f1":
                d_out[k] = [v.f1]
        return pd.DataFrame.from_dict(d_out)
    
    
class tf_act_cross_peaks_prod_acc(tf_act_cross_peaks_alt):
    def __init__(self, gt, metrics, jaspar_motifs, chip_bulk=None, pred_acc=None):
        super().__init__(gt, metrics, jaspar_motifs, chip_bulk)
        if not pred_acc is None:
            self.pred_acc = pred_acc.loc[self.peak_shared, :]
        pass
    
    def logReg_all(self, downsample_frac=1., trainOn="chip_bulk", mode="logReg", **model_kwargs):
        self.logRegs_all = {}
        y_pred = {}
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
                                
                logreg = tf_act_Reg(X=X, 
                                    Y=Y, 
                                    Y_test=self.gt.loc[:,tf_ct].to_numpy(),
                                    downsample_frac=downsample_frac,
                                    label=f"{tf}:{mm}_{ct}", 
                                    mode=mode,
                                    **model_kwargs)
                
                self.logRegs_all[f"{tf}:{mm}_{ct}"] = logreg 
                y_pred[f"{tf}:{mm}_{ct}"] = logreg.y_pred * self.pred_acc.loc[:,ct]
        self.y_pred = pd.DataFrame(y_pred, index=self.gt.index)
        
        return
    
    def auprcs_to_csv(self, metric="auprc"):
        return super().auprcs_to_csv(metric)


from sklearn.metrics import average_precision_score
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")

def auprc_cross_cells(y_true, y_score):
    """(n_peak, n_ct)"""
    auprcs = np.zeros((y_true.shape[0]))
    for i in tqdm(range(y_true.shape[0])):
        auprcs[i] = average_precision_score(y_true=y_true[i,:], y_score=y_score[i,:])
    return auprcs

def get_auprc_cross_tfs(bm_to_plot_dict, df_gt):
    y_pred_auprc = {}
    for k in bm_to_plot_dict.keys():
        bm_model = bm_to_plot_dict[k]
        df_tmp = bm_model.y_pred.T
        df_tmp['tf_ct'] = df_tmp.index.str.split("_").str[0].str.split(":").str[0] + "_" + df_tmp.index.str.split("_").str[1]
        df_tmp = df_tmp.groupby("tf_ct").max().T.loc[df_gt.index, df_gt.columns]
        auprc = auprc_cross_cells(df_gt.to_numpy(), df_tmp.to_numpy())
        y_pred_auprc[k] = auprc 
    return pd.DataFrame(y_pred_auprc, index=df_gt.index)

def get_auprc_cross_peaks(bm_to_plot_dict, df_gt):
    y_pred_auprc = {}
    for k in bm_to_plot_dict.keys():
        bm_model = bm_to_plot_dict[k]
        df_tmp = bm_model.y_pred.T          ## after prod
        df_tmp['tf_ct'] = df_tmp.index.str.split("_").str[0].str.split(":").str[0] + "_" + df_tmp.index.str.split("_").str[1]
        df_tmp = df_tmp.groupby("tf_ct").max().T.loc[df_gt.index, df_gt.columns]        #(peak, tf_ct)
        
        auprc = auprc_cross_cells(df_gt.to_numpy().T, df_tmp.to_numpy().T)
        y_pred_auprc[k] = auprc 
    return pd.DataFrame(y_pred_auprc, index=df_gt.columns)
    
    
    
    
    
    