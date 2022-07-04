# -*- coding: utf-8 -*-
'''
This code will do the necessary preprocessing steps on the Mouse Cortex and Brain Large data sets.

'''

import numpy as np
import pandas as pd
import torch
import h5py

from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def Zeisel_data(file_dir= "/home/longlab/Data/Thesis/Data/expression_mRNA_17-Aug-2014.txt", 
           n_genes = 558):
    '''
    This function will perform the pre-processing steps for the gold standard Zeisel data set. 
    
    This steps are: 
    1. exctracting the labels of the cell types from the data
    2. choosing the genes that are transcribed in more than 25 cells 
    3. Selecting the 558 genes with the highest Variance in the remaining genes from the previous step
    4. Performing random permutation of the genes 
    
    Parameters
    ----------
    file_dir : str 
        The path to the .csv file.
        
    n_genes : int 
        Number of the high variable genes that should be selected.
    
    Returns
    -------
    y : Pytorch ndarray
        The permuted dataset with n_genes and 3005 cells
    
    labels : str
        The true labels (cell types)
    '''
    np.random.seed(197)
    
    df = pd.read_csv(file_dir, delimiter= '\t', low_memory=False)
    # Groups of cells, i.e. labels
    Groups = df.iloc[7,]
    # print(Groups.value_counts())
    Groups = Groups.values[2:]
    _ , labels = np.unique(Groups, return_inverse=True)
    
    df1 = df.iloc[10:].copy()
    df1.drop(['tissue', 'Unnamed: 0'], inplace=True, axis=1)
    df1 = df1.astype(float)
    
    rows = np.count_nonzero(df1, axis=1) > 0 # choosing genes that are transcribed in more than 25 cells
    df1 = df1[np.ndarray.tolist(rows)]
    # df2 += 1
    data = df1.values
    # data = np.log10(data)  # log transforming the data
    
    # permutation
    # selecting the number of genes, i.e. number of features
    
    # sorting the data based on the variance
    rows = np.argsort(np.var(data, axis = 1)*-1) 
    data = data[rows,:]
    data = data[0:n_genes, :] # choosing high variable genes 
    
    p = np.random.permutation(data.shape[0])
    data[:,:] = data[p,:]
    
    p = np.random.permutation(data.shape[1])
    data[:,:] = data[:,p]
    
    labels = labels[p]
    
    y = data.T #whole data set
    y = torch.from_numpy(y)
    
    

    return y, labels


class Brain_Large(Dataset):
    
    '''
    The Dataset class with necessary pre-processing steps for the Large brain dataset. 
    
    The variance of the genes for a sub-sample of 10^5 cells will be calculated 
    and the high variable genes (720 by default) will be selected.
    
    
    The Large brain dataset can be downloaded from the following url:
        "http://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5"
           
    The data is in HDF5 format which can be easily accessed and processed with the `h5py` library.
    
    The data contains: 
        
        barcodes: This contains information of the batch number which can be used for batch correction.
        
        `gene_names` contains the gene names.
        `genes` contains the Ensembl Gene id such as: 'ENSMUSG00000089699'
        `data` is an array containing all the non zero elements of the sparse matrix 
        `indices` is an array mapping each element in `data` to its column in the sparse matrix.
        `indptr` maps the elements of data and indices to the rows of the sparse matrix.
        `shape` the dimension of the sparse matrix
        
        For more info please visit "https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr"
        and "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html"
    
    Attributes
    ----------
    file_dir : str
        The directory of the HDF5 file
    n_select_genes : int
        Number of high variable genes to be selected in the pre-processing step.
    n_genes 
        Total number of genes in the data set
    n_cells
        Total number of cells
    selected_genes
        The indices of selected high variable genes.

     '''
    def __init__(self, file_dir, n_sub_samples = 10**5, n_select_genes = 720):
        
        '''
        Initialize the Brain_Large Dataset class
        
        It will performs the necessary pre-processing steps for selecting the high variable genes 
        and set the indices of them.
        
        
        Parameters
        ----------
        file_dir: str 
            The directory of the HDF5 file which should be provided by the user.
            
        n_sub_samples: int
            Number of samples (cells) in the downsampled matrix. Default: 10^5
        n_select_genes
            Number of the high variable genes to be selected in the pre-processing step. Default: 750
        
        '''
        
        self.file_dir = file_dir
        self.n_select_genes = n_select_genes
        
        with h5py.File(self.file_dir, 'r') as f:
            
            data = f['mm10']
            self.n_genes, self.n_cells = data['shape']
            
            # sub-sampling for choosing high variable genes
            
            indptr_sub_samp = data['indptr'][:n_sub_samples]
            last_indx = indptr_sub_samp[-1]
            
            sub_sampled = csc_matrix((data['data'][:last_indx], 
                                 data['indices'][:last_indx], 
                                 indptr_sub_samp),
                                shape=(self.n_genes, n_sub_samples-1))
            
            scale = StandardScaler(with_mean = False)
            scale.fit(sub_sampled.T)
            
            # choosing high variable genes
            self.selected_genes = np.argsort(scale.var_*-1)[:self.n_select_genes]
            
    def __len__(self):
        return self.n_cells
        
    def __getitem__(self, index):
        
        with h5py.File(self.file_dir, 'r') as f:
            data = f['mm10']
            
            indptr_sub_samp = data['indptr'][index:(index+1)+1]
            first_indx = indptr_sub_samp[0]
            last_indx = indptr_sub_samp[-1]
            indptr_sub_samp = (indptr_sub_samp - first_indx).astype(np.int32)
            matrix_batch = csc_matrix((data['data'][first_indx:last_indx], 
                             data['indices'][first_indx:last_indx], 
                             indptr_sub_samp),
                            shape=(self.n_genes,1))
        
        matrix_batch = matrix_batch.toarray().T[:, self.selected_genes]
        matrix_batch = torch.tensor(matrix_batch, dtype=torch.float32)
        return matrix_batch
            
#     Another thing is that we will not have enough memory to train the zinb on big datasets
#     I think we should do it based on batches => how to do it? have a fixed big matrix 
#     and then train for a portion of that matrix? 

# =============================================================================
# def Brain_large(base_path = "/home/longlab/Data/Thesis/Data/", 
#            n_select_genes = 720):
#     
#     
#     np.random.seed(197)
#     
#     with h5py.File("1M_neurons_filtered_gene_bc_matrices_h5.h5", 'r') as f:
#         
#         data = f['mm10']
#         n_genes, n_cells = data['shape']
#         
#         # sub-sampling for choosing high variable genes
#         n_sub_samples = 10**5
#         indptr_sub_samp = data['indptr'][:n_sub_samples]
#         last_indx = indptr_sub_samp[-1]
#         
#         sub_sampled = csc_matrix((data['data'][:last_indx], 
#                              data['indices'][:last_indx], 
#                              indptr_sub_samp),
#                             shape=(n_genes, n_sub_samples-1))
#         
#         scale = StandardScaler(with_mean = False)
#         scale.fit(sub_sampled.T)
#         
#         # choosing high variable genes
#         selected_genes = np.argsort(scale.var_*-1)[:n_select_genes]
#         
#         del sub_sampled, scale
# 
#         n_sub_samples = n_cells/17
# 
#         
#         for i in range(17): 
#             
#             indptr_sub_samp = data['indptr'][(i*n_sub_samples):((i+1)*n_sub_samples+1)]
#             first_indx = indptr_sub_samp[0]
#             last_indx = indptr_sub_samp[-1]
#             
#             matrix_batch = csc_matrix((data['data'][first_indx:last_indx], 
#                              data['indices'][first_indx:last_indx], 
#                              indptr_sub_samp),
#                             shape=(n_sub_samples, n_genes))[:, selected_genes]
#             
#             matrix = vstack([matrix, matrix_batch])
#             
#         
#         
#         
# # =============================================================================
# #         barcodes = f["mm10"]["barcodes"][...]
# #         batch_id = np.array([batch[int(x.split("-")[-1])-1] for x in barcodes])
# # =============================================================================
#         
#         
#     y = matrix.T #whole data set
#     y = torch.from_numpy(y)
#    
#     
# 
#     return y, labels
# =============================================================================
