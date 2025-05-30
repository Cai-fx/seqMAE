import numpy as np 
import jax 
import h5py 
import random
from scipy import sparse 
import anndata 
import os 
import time 
import sys 
import pysam
from tqdm import tqdm 

def pos_2_seq(mat):
    def map_nt(x : int):
        if x==0:
            return "A"
        if x==1:
            return "C"
        if x==2:
            return "G"
        if x==3:
            return "T"
        return 
    seq = "" 
    for _nt in mat:
        seq += map_nt(_nt)
    return seq 

def write_fasta(filename:str, seqs:np.ndarray, seq_names:np.ndarray):
    with open(filename, "w+") as f:
        for i, _seq_mat in tqdm(enumerate(seqs)):
            f.write(f">{seq_names[i]}\n")
            _seq = pos_2_seq(_seq_mat)
            f.write(_seq)
            f.write("\n")
    return 

def split_cells(n_cell:int, 
                write_dest = "./data/rna_data/cell_splits.h5", 
                rng=jax.random.PRNGKey(547547)):
    cell_ids = np.arange(n_cell)
    n_train = int(n_cell*0.7)
    n_val = int(n_cell*0.1)
    n_test = n_cell - n_train - n_val 
    rng1, rng2 = jax.random.split(rng)
    train_idx = np.sort(jax.random.choice(key=rng1, a=cell_ids, shape=(n_train,), replace=False))
    val_idx = np.sort(jax.random.choice(key=rng2, a=np.setdiff1d(cell_ids, train_idx), shape=(n_val,), replace=False))
    test_idx = np.sort(np.setdiff1d(np.setdiff1d(cell_ids, train_idx), val_idx))

    hf = h5py.File(write_dest, 'w')

    hf.create_dataset('train_ids', data=train_idx)
    hf.create_dataset('val_ids', data=val_idx)
    hf.create_dataset('test_ids', data=test_idx)

    hf.close()
    return

def dna_1hot_2vec(seq, seq_len=None):
    """ from scbasset 
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.

    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, ), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] =  random.randint(0, 3)
    return seq_code



def split_train_test_val(ids, seed=10, train_ratio=0.9):
    np.random.seed(seed)
    test_val_ids = np.random.choice(
        ids,
        int(len(ids) * (1 - train_ratio)),
        replace=False,
    )
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(
        test_val_ids,
        int(len(test_val_ids) / 2),
        replace=False,
    )
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    return train_ids, test_ids, val_ids

def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i,0]
        start = int(input_bed.iloc[i,1])
        end = int(input_bed.iloc[i,2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords


def make_h5_sparse(tmp_ad, h5_name, input_fasta, seq_len=1344, batch_size=1000):
    ## batch_size: how many peaks to process at a time
    ## tmp_ad.var must have columns chr, start, end
    
    t0 = time.time()
    
    m = tmp_ad.X
    m = m.tocoo().transpose().tocsr()
    n_peaks = tmp_ad.shape[1]
    bed_df = tmp_ad.var.loc[:,['chr','start','end']] # bed file
    bed_df.index = np.arange(bed_df.shape[0])
    n_batch = int(np.floor(n_peaks/batch_size))
    batches = np.array_split(np.arange(n_peaks), n_batch) # split all peaks to process in batches
    
    ### create h5 file
    # X is a matrix of n_peaks * 1344
    f = h5py.File(h5_name, "w")
    
    ds_X = f.create_dataset(
        "X",
        (n_peaks, seq_len),
        dtype="int8",
    )

    # save to h5 file
    for i in range(len(batches)):
        
        idx = batches[i]
        # write X to h5 file
        seqs_dna,_ = make_bed_seqs_from_df(
            bed_df.iloc[idx,:],
            fasta_file=input_fasta,
            seq_len=seq_len,
        )
        dna_array_dense = [dna_1hot_2vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense
            
        t1 = time.time()
        total = t1-t0
        print('process %d peaks takes %.1f s' %(i*batch_size, total))
    
    f.close()

def preprocess_seqs(input_ad, input_fasta, output_path):
    
    ad = anndata.read_h5ad(input_ad)

    os.makedirs(output_path, exist_ok=True)
    seq_len = 1344

    # save anndata
    ad.write('%s/ad.h5ad'%output_path)
    print('successful writing h5ad file.')

    # save peak bed file
    ad.var.loc[:,['chr','start','end']].to_csv('%s/peaks.bed'%output_path, sep='\t', header=False, index=False)
    print('successful writing bed file.')

    # save train, test, val splits
    train_ids, test_ids, val_ids = split_train_test_val(np.arange(ad.shape[1]))
    f = h5py.File('%s/splits.h5'%output_path, "w")
    f.create_dataset("train_ids", data=train_ids)
    f.create_dataset("test_ids", data=test_ids)
    f.create_dataset("val_ids", data=val_ids)
    f.close()
    print('successful writing split file.')

    # save labels (ad.X)
    m = ad.X.tocoo().transpose().tocsr()
    m_train = m[train_ids,:]
    m_val = m[val_ids,:]
    m_test = m[test_ids,:]
    sparse.save_npz('%s/m_train.npz'%output_path, m_train, compressed=False)
    sparse.save_npz('%s/m_val.npz'%output_path, m_val, compressed=False)
    sparse.save_npz('%s/m_test.npz'%output_path, m_test, compressed=False)
    print('successful writing sparse m.')

    # save sequence h5 file
    ad_train = ad[:,train_ids]
    ad_test = ad[:,test_ids]
    ad_val = ad[:,val_ids]
    make_h5_sparse(ad, '%s/all_seqs.h5'%output_path, input_fasta)
    make_h5_sparse(ad_train, '%s/train_seqs.h5'%output_path, input_fasta)
    make_h5_sparse(ad_test, '%s/test_seqs.h5'%output_path, input_fasta)
    make_h5_sparse(ad_val, '%s/val_seqs.h5'%output_path, input_fasta)

