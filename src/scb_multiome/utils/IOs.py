import numpy as np
import scipy.sparse as sparse
import h5py
import pickle
import pickle5
from functools import reduce 
import operator
from flax.core.frozen_dict import freeze, unfreeze
import pandas as pd 

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)



class seq_generator:
    def __init__(self, file, m):
        self.file = file # h5 file for sequence
        self.m = m # csr matrix, (seqs, cells)
        self.n_cells = m.shape[1]
        self.ones = np.ones(1344)
        self.rows = np.arange(1344)
        pass

class seq_acc_generator(seq_generator):
    def __init__(self, file, m):
        super().__init__(file, m)
        
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            for i in range(X.shape[0]):
                x = X[i]
                x_tf = sparse.coo_matrix((self.ones, (self.rows, x)), 
                                               shape=(1344, 4), 
                                               dtype='float32').toarray()
                y = self.m.indices[self.m.indptr[i]:self.m.indptr[i+1]]
                y_tf = np.zeros(self.n_cells, dtype='float32')
                y_tf[y] = 1.
                yield x_tf, y_tf

class seq_rna_generator(seq_generator):
    def __init__(self, file, m):
        super().__init__(file, m)
    
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            for i in range(X.shape[0]):
                x = X[i]
                x_tf = sparse.coo_matrix((self.ones, (self.rows, x)), 
                                               shape=(1344, 4), 
                                               dtype='float32').toarray()
                y_tf = self.m[i,:]
                yield x_tf, y_tf




def load_params(init_params, path=f"", load_keys=["params_all"], exclude_list=[]):
    with open(path, 'rb') as handle: 
        params = pickle5.load(handle)
    
    params =  getFromDict(params, load_keys)
    
    init_params, params = unfreeze(init_params), unfreeze(params)
    init_params = update_nested_dict(params, init_params, exclude_list=exclude_list)
    return freeze(init_params)



def update_nested_dict(d1, d2, exclude_list=[]):
    '''assign d2[key] value as d1[key]'''
    for key, value in d1.items():
        if (key in d2 ) and not (key in exclude_list):
            print(f"initializing field: {key}")
            if isinstance(value, dict) and isinstance(d2[key], dict):
                update_nested_dict(value, d2[key], exclude_list=exclude_list)
            else:
                d2[key] = value
    return d2
