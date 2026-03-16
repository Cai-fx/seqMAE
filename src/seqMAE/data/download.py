import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests


REMOTE_URL = "https://zenodo.org/records/16810645/files"

def download_base(file_url:str, force_download=False, save_path=""):
    if save_path:
        dest = Path(save_path)
    else:
        raise Exception("empty save path")
    
    filename = file_url.split('/')[-1] 
    
    if not force_download and (dest/filename).exists():
        return dest
    
    os.makedirs(dest, exist_ok=True)
    
    
    # Streaming download with progress bar
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(dest/filename, 'wb') as f:
            with tqdm.wrapattr(
                f, 
                "write",
                total=total_size,
                desc=f"Downloading {file_url.split('/')[-1]}"
            ) as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)
    
    return dest/filename


def download_jaspar_motifs(force_download=False, save_path=""):
    """
    _mode = "draft" or "published"
    """
    file_url = f"{REMOTE_URL}/20230424043428_JASPAR2022_combined_matrices_2028_meme.txt"
    return download_base(file_url=file_url, force_download=force_download, save_path=save_path)

def download_encode_chip(force_download=False, save_path=""):
    file_url = f"{REMOTE_URL}/ENCODE_ChIP.tar.gz"
    return download_base(file_url=file_url, force_download=force_download, save_path=save_path)
    
def download_pbmc_supp(force_download=False, save_path=""):
    file_url = f"{REMOTE_URL}/pbmc_data_for_zenodo.tar.gz"
    return download_base(file_url=file_url, force_download=force_download, save_path=save_path)

def download_pbmc_raw(force_download=False, save_path=""):
    file_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
    return download_base(file_url=file_url, force_download=force_download, save_path=save_path)
