# -*- coding: utf-8 -*-
'''
This code will do the necessary preprocessing steps on the Mouse Cortex dataset
This steps are: 
    1. exctracting the labels of the cell types from the data
    2. choosing the genes that are transcribed in more than 25 cells 
    3. Selecting the 558 genes with the highest Variance in the remaining genes from the previous step
    4. Performing random permutation of the genes 
'''

import numpy as np
import pandas as pd
import torch


if __name__ == '__main__':
    
    base_dir = "/home/longlab/Data/Thesis/Data/"
    
    np.random.seed(197)
    
    df = pd.read_csv(base_dir + 'Zeisel.zip', compression='zip', delimiter= '\t', low_memory=False)
    # Groups of cells, i.e. labels
    Groups = df.iloc[0,]
    # print(Groups.value_counts())
    Groups = Groups.values[2:]
    df1 = df.iloc[10:].copy()
    df1.drop(['tissue', 'Unnamed: 0'], inplace=True, axis=1)
    df1 = df1.astype(float)
    
    rows = np.count_nonzero(df1, axis=1) > 0 # choosing genes that are transcribed in more than 25 cells
    df1 = df1[np.ndarray.tolist(rows)]
    # df2 += 1
    data = df1.values
    # data = np.log10(data)  # log transforming the data
    
    # permutation
    n_genes = 558 # selecting the number of genes, i.e. number of features
    
    rows = np.argsort(np.var(data, axis = 1)*-1) # sorting the data based on the variance
    data = data[rows,:]
    data = data[0:n_genes, :] # choosing high variable genes 
    
    p = np.random.permutation(data.shape[0])
    data[:,:] = data[p,:]
    
    p = np.random.permutation(data.shape[1])
    data[:,:] = data[:,p]
    
    Groups = Groups[p]
    
    y = data.T #whole data set
    y = torch.from_numpy(y)