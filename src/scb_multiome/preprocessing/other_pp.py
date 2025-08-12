import numpy as np
import pandas as pd
import re
import pybedtools 
import glob 
from tqdm import tqdm 
import anndata
from scipy.stats import spearmanr 
from joblib import Parallel, delayed

class MOTIF:
    def __init__(self, name, tf=None, width=0) -> None:
        self.name=name
        self.tf = tf
        self.width = width
        self.pwm_couter=0
        self.pwm = np.zeros((self.width, 4))
        pass
    
    def read_pwm(self, l):
        w = list(filter(None, re.split(r'[\s]+', l)))
        arr = np.array(w, dtype=float)
        self.pwm[self.pwm_couter, :] = arr
        self.pwm_couter +=1
        return


def read_JASPAR_pwms(meme_file):
    start_line_num, width = 0, 0
    motifs = []
    with open(meme_file, 'r') as handle:
        for i, line in enumerate(handle):
            match_motif = re.search("^MOTIF", line)
            start_pwm = re.search("^letter-probability matrix:", line)
            end_pwm = re.search("^URL", line)
            
            if match_motif:
                keys = re.split(r'[\s]', line)
                motif = keys[1]
                tf = keys[2].split('.')[-1]
            
            if start_pwm:
                width = re.search(r'w= [(0-9)]+', line)
                width = int(line[width.start()+3: width.end()])
                mot = MOTIF(motif, tf, width)
                start_line_num = i
            
            pwm_lines = i>start_line_num and i<=start_line_num+width
            if pwm_lines:
                mot.read_pwm(line)
            
            if end_pwm:
                motifs.append(mot)

    motif_name = [mot.name for mot in motifs]
    tf_name = [mot.tf for mot in motifs]
    pwm = [mot.pwm for mot in motifs]
    df = pd.DataFrame.from_dict({'motif':motif_name ,'tf': tf_name, 'pwm': pwm})
    df['tf']=df['tf'].str.split('::')
    df = df.explode('tf')
    df = df.sort_values('tf').reset_index(drop=True)
    return df 



def pad_pwm_df(df, pad_val=0):
    max_len = list(map(lambda x: x.shape[0], df['pwm']))
    max_len = np.max(max_len)
    print(f"max_len={max_len}")
    pwms = list(map(lambda x: np.pad(x, ((0,max_len-x.shape[0]), (0,0)), constant_values=pad_val),  df['pwm']))
    df['pwm'] = pwms
    pwms = np.asarray(pwms)
    return pwms




def chip_atac_overlap(ad_atac:anndata.AnnData, 
                      encode_dir="", 
                      meta_data:pd.DataFrame=None,
                      groupkey="Target of assay",
                      bed_files_postfix="bed"):
    atac_peaks = ad_atac.var[["chr", "start", "end"]]
    atac_peaks.columns = ["chrom", 'start', 'stop']
    atac_bed = pybedtools.BedTool.from_dataframe(atac_peaks)
    bed_files = glob.glob(f"{encode_dir}/*.{bed_files_postfix}")
    
    print("num of bed files", len(bed_files))
    
    encode_overlap = pd.DataFrame([], index=ad_atac.var.index)

    for f in tqdm(bed_files):
        f_accession = f.split("/")[-1].split(".")[0]
        aa = pybedtools.BedTool(f)
        overlap_df = pd.read_table((atac_bed+aa).fn, 
                                names=['chrom', 'start', 'stop', 'name', 'score', 'strand', 'a', 'b', 'c', 'd'])
        overlap_df.index = overlap_df["chrom"]+":"+overlap_df["start"].astype("str")+"-"+overlap_df["stop"].astype("str")
        overlap_df = overlap_df[["chrom", "start", "stop"]]
        encode_overlap[f_accession] = atac_peaks.index.isin(overlap_df.index)
        
    ## aggregate peaks by key 
    df_allinfo = pd.concat([encode_overlap, meta_data.loc[encode_overlap.columns].T])
    count_by_tf = df_allinfo.T.groupby(groupkey).sum()
    count_by_tf = count_by_tf[encode_overlap.index].astype(int)
    return (count_by_tf>0)

def _scc(a, b):
    return spearmanr(a, b)[0]

def celltype_specific_spearmanr(ad_atac, 
                                ad_rna, 
                                tfs_cts, 
                                atac_layer="pvi",
                                gene_key = "gene_symbols",
                                celltype_key="celltype",
                                n_jobs=16):
    celltype_specific_SpearmanR = pd.DataFrame(index=ad_atac.var.index)
    for tf_ct in tfs_cts:
        tf, ct = tf_ct.split("_")[0], tf_ct.split("_")[1]
        tf_idx = np.where(ad_rna.var[gene_key]==tf)[0][0]
        print("tf_idx = ", tf_idx)
        cell_idx = np.where(ad_rna.obs[celltype_key] == ct)[0]
        print(f"number of cells in {ct} = {len(cell_idx)}")
        rna_expr = ad_rna.X[cell_idx, tf_idx]
        atac_expr = ad_atac.layers[atac_layer][cell_idx,:].T

        res = Parallel(n_jobs=n_jobs,
                backend="multiprocessing")(
                    delayed(_scc)(rna_expr, _atac_expr) for _atac_expr in tqdm(atac_expr)
                    )

        res = np.asarray(res)
        celltype_specific_SpearmanR[f"{tf}_{ct}"] = res 
    return celltype_specific_SpearmanR


def read_fimo_res(fimo_dir, peak_idx:pd.Series, jaspar_motifs:pd.DataFrame, f_name="fimo.tsv",
                  seq_name_key="sequence_name"):
    '''reads fimo results of positive peaks, return those with p-val <0.05, pad others with 1'''
    act_pval = []
    for motif_id in jaspar_motifs['motif'].unique():
        tf_name = jaspar_motifs[jaspar_motifs["motif"]==motif_id]["tf"].values[0]
        file_name = f"{fimo_dir}/{motif_id}/{f_name}"
        skiprows = sum(1 for line in open(file_name) if line.startswith('#')) 
        fimo_res = pd.read_csv(file_name, sep="\t", skipfooter=skiprows)
        if len(fimo_res)>0:
            fimo_res = fimo_res[(fimo_res['p-value']<0.05) 
                                #& (fimo_res['q-value']<0.05)
                                ]                      # filter p & q val 
            fimo_res.index = fimo_res[seq_name_key]
            fimo_res = fimo_res.drop_duplicates(subset=seq_name_key, keep='first')                                                     # drop multiple indices in single sequence
            act_scores = fimo_res.reindex(peak_idx, fill_value=1.)[["p-value"]]
            act_scores = act_scores.rename({"p-value": f"{tf_name}:{motif_id}"}, axis=1)
            act_pval.append(act_scores)
    act_pval = pd.concat(act_pval, axis=1)
    act_pval = np.log(act_pval)                  
    return act_pval