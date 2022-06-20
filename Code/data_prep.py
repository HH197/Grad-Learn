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


def Zeisel_data(base_path = "/home/longlab/Data/Thesis/Data/", 
           n_genes = 558):
    
    np.random.seed(197)
    
    df = pd.read_csv(base_path + 'expression_mRNA_17-Aug-2014.txt', 
                     delimiter= '\t', low_memory=False)
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