
'''
This code will do the necessary pre-processing steps on single-cell RNA-seq datasets.
@author: HH197
'''

import numpy as np
import pandas as pd
import torch
import h5py
import loompy

from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import Sampler



class CORTEX(Dataset):
    
    ''' 
    The Dataset class with necessary pre-processing steps for the gold standard
    Zeisel data set.
    
    '''
    
    def __init__ (self, 
                  file_dir= "/home/longlab/Data/Thesis/Data/expression_mRNA_17-Aug-2014.txt", 
                  n_genes = 558):
    
        '''
        
        Performs the pre-processing steps which are:
        
        1. exctracting the labels of the cell types from the data
        2. choosing the genes that are transcribed in more than 25 cells 
        3. Selecting the 558 genes with the highest Variance in the remaining genes from the previous step
        4. Performing random permutation of the genes 
        
        Attributes
        ----------
        file_dir : str 
            The path to the .csv file.
        n_genes : int 
            Number of the high variable genes that should be selected.
        n_cells
            Total number of cells.
        data: torch Tensor
            The data.
        labels: torch Tensor
            The labels.
        '''
        
        self.file_dir = file_dir
        self.n_genes = n_genes
        
        df = pd.read_csv(self.file_dir, delimiter= '\t', low_memory=False)
        
        # Groups of cells, i.e. labels
        Groups = df.iloc[7,]
        
        Groups = Groups.values[2:]
        _ , labels = np.unique(Groups, return_inverse=True)
        
        df1 = df.iloc[10:].copy()
        df1.drop(['tissue', 'Unnamed: 0'], inplace=True, axis=1)
        df1 = df1.astype(float)
        
        # choosing genes that are transcribed in more than 25 cells
        rows = np.count_nonzero(df1, axis=1) > 0 
        df1 = df1[np.ndarray.tolist(rows)]
        
        data = df1.values
        
        # sorting the data based on the variance
        rows = np.argsort(np.var(data, axis = 1)*-1) 
        data = data[rows,:]
        data = data[0:self.n_genes, :] # choosing high variable genes 
        
        #permutation on features (they were sorted)
        np.random.seed(197)
        p = np.random.permutation(data.shape[0])
        data = data[p,:]
        
        y = data.T
        
        self.n_cells = y.shape[0]
        self.data = torch.from_numpy(y)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return self.n_cells
    
    def __getitem__ (self, index):
        return self.data[index], self.labels[index]


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
    n_genes : int
        Total number of genes in the data set
    n_cells : int
        Total number of cells
    selected_genes : ndarray
        The indices of selected high variable genes.
    low_memory : boolean
        Whether perform data loading with memory efficiency or not.
    matrix : scipy csc_matrix
        If `low_memory` is False, the whole data otherwise `None`.
        
     '''
    def __init__(self, 
                 file_dir = "/home/longlab/Data/Thesis/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5",
                 n_sub_samples = 10**5, 
                 n_select_genes = 720, 
                 low_memory = True):
        
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
        n_select_genes: int
            Number of the high variable genes to be selected in the pre-processing
            step. Default: 750
        low_memory: Boolean
            If `False`, the whole data will be loaded into memory (high amount of
            memory required); `True` by default.
        
        '''
        
        self.file_dir = file_dir
        self.n_select_genes = n_select_genes
        self.low_memory = low_memory
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
        
        if low_memory == False: 
            
            del sub_sampled, scale, indptr_sub_samp
            print('Loading the whole dataset')
            
            
            n_iters = int(self.n_cells / 10**5) + (self.n_cells % 10**5 > 0)
            
            with h5py.File(self.file_dir, 'r') as f:
                
                data = f['mm10']
                index_partitioner = data["indptr"][...]
                
                # The following code is from the scvi package
                for i in range(n_iters):
                    print(f"Reading batch {i+1}")
                    index_partitioner_batch = index_partitioner[
                        (i * 10**5) : ((1 + i) * 10**5 + 1)]
                    first_index_batch = index_partitioner_batch[0]
                    last_index_batch = index_partitioner_batch[-1]
                    index_partitioner_batch = (index_partitioner_batch - first_index_batch).astype(np.int32)
                    n_cells_batch = len(index_partitioner_batch) - 1
                    data_batch = data["data"][first_index_batch:last_index_batch].astype(
                        np.float32)
                    indices_batch = data["indices"][first_index_batch:last_index_batch].astype(
                        np.int32)
                    matrix_batch = csr_matrix(
                        (data_batch, indices_batch, index_partitioner_batch),
                        shape=(n_cells_batch, self.n_genes),
                    )[:, self.selected_genes]
                    # stack on the fly to limit RAM usage
                    if i == 0:
                        matrix = matrix_batch
                    else:
                        matrix = vstack([matrix, matrix_batch])
                self.matrix = matrix

        else: 
            self.matrix = None

                
    def __len__(self):
        return self.n_cells
        
    def __getitem__(self, index):
        
        if self.low_memory:
                
            with h5py.File(self.file_dir, 'r') as f:
                
                data = f['mm10']
                
                # try to initalize the data in init and then use data in the getitem? 
                
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
            return matrix_batch, index
        else:
            return torch.tensor(self.matrix[index].A, dtype=torch.float32), index
        
class Indice_Sampler(Sampler):
    
    '''
    This is a sampler for Pytorch ``DataLoader`` in which it will sample from the 
    ``Dataset`` with indices of mask.
    
    Parameters
    ----------
    
    mask: ``torch.tensor`` 
        The indices of the samples.
    
    
    Example
    ----------
        >>> from torch.utils.data import DataLoader
        >>> brain = Brain_Large()
        >>> dataloader = torch.utils.data.DataLoader(brain, 
                                                     batch_size=64, 
                                                     shuffle=True)
        >>> a, b = next(iter(dataloader))
        >>> trainloader_sampler1 = torch.utils.data.DataLoader(brain,
                                                           batch_size=64,
                                                           sampler = Brain_Large_Sampler(b),
                                                           shuffle=False)
        >>> c, d = next(iter(trainloader_sampler1))
        
    '''
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (i for i in self.mask)

    def __len__(self):
        return len(self.mask)


class RETINA(Dataset):
    
    '''
    The Dataset class with necessary pre-processing steps for the RETINA dataset. 
    
    The data provided by Lopez et al. (scvi) is used which can be 
    downloaded from: https://github.com/YosefLab/scVI-data/raw/master/retina.loom
    The data is in loompy format (http://linnarssonlab.org/loompy/index.html)
    
    The original dataset is available under the GEO accession number GSE81904.
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81904
    
    The webpage consists of the raw matrix, BAM files, Rdata files, and R code. 
    To perform the pre-processing steps and analyses of the original paper, 
    one can use the following github repository: 
        https://github.com/broadinstitute/BipolarCell2016
    
    
    Note: 
        Since the data is not very large (approximately 2GB in memory), the class will
        load the whole data into memory.
    
    
    Attributes
    ----------
    file_dir : str
        The directory of the HDF5 file.
        
    n_genes : int 
        Total number of genes in the data set.
    
    n_cells : int
        Total number of cells.
    
    batchID : ndarray
        One dimensional array of the Batch Ids of cells.
    
    cluster : ndarray
        One dimensional array of the assigned cluster numbers by the authors of 
        the original paper.
     '''
    def __init__(self, 
                 file_dir = "/home/longlab/Data/Thesis/Data/retina.loom"):
        
        '''
        Initialize the RETINA Dataset class.
        
        It seperates the raw count data, the cluster numbers, and batches. 
        
        Parameters
        ----------
        file_dir: str 
            The directory of the HDF5 file which should be provided by the user.
            
        '''
        
        self.file_dir = file_dir
        
        with loompy.connect(self.file_dir) as ds:
            
            self.data = torch.from_numpy(ds[:,:].T).float()
            self.n_genes, self.n_cells = ds.shape
            self.batchID = ds.ca['BatchID'].ravel()
            self.cluster = ds.ca['ClusterID'].ravel()
            
    def __len__(self):
        return self.n_cells
        
    def __getitem__(self, index):
        
        matrix_batch = self.data[index]
        return matrix_batch, index

