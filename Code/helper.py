#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:19:08 2022

@author: hh197

This will contain all of the helper functions. 
"""

import seaborn as sn
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np

def kmeans(data, kmeans_kwargs = {"init": "random", 
                                   "n_init": 50, 
                                   "max_iter": 400, 
                                   "random_state": 197}):
        
    """Performing K-means on the encoded data"""
    
    sil_coeff = []
    
    
    
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        # score = kmeans.inertia_
        score = metrics.silhouette_score(data, kmeans.labels_)
        sil_coeff.append(score)
        
    return sil_coeff

def plot_si(sil_coeff):
        
    """Silhouette plot"""
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 10), sil_coeff)
    plt.xticks(range(2, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    # plt.savefig(base_dir + 'test1.png')


def measure_q(data, Groups= None, n_clusters=6,  
              
              kmeans_kwargs = {"init": "random", 
                               "n_init": 50, 
                               "max_iter": 400, 
                               "random_state": 197}):
    
    """Measuring the quality of clustering using NMI and confusion matrix"""
    
    kmeans = KMeans(n_clusters, **kmeans_kwargs)
    kmeans.fit(data)
    
    
    Groups = np.ndarray.astype(Groups, np.int)
    NMI = metrics.cluster.normalized_mutual_info_score(Groups, kmeans.labels_)
    print(f'The NMI score is: {NMI}')
    CM = metrics.confusion_matrix(Groups, kmeans.labels_)
    
    df_cm = DataFrame(CM, range(1, CM.shape[0]+1), range(1, CM.shape[1]+1))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 5}) # font size
    
    plt.show()

def corrupting(data, p = 0.10, method = 'Uniform', percentage = 0.10):
    
    '''
    Adopted from the "Deep Generative modeling for transcriptomics data"
    
    This function will corrupt  (adding noise or dropouts) the datasets for
    imputation benchmarking. 
    
    Two different approaches for data corruption: 
        1. Uniform zero introduction: Randomly selected a percentage of the nonzero 
        entries and multiplied the entry n with a Ber(0.9) random variable. 
        2. Binomial data corruption: Randomly selected a percentage of the matrix and
replaced an entry n with a Bin(n, 0.2) random variable.


    Parameters
    ----------
    data : numpy ndarray 
        The data.
        
    p : float >= 0 and <=1
        The probability of success in Bernoulli or Binomial distribution.
        
    method: str 
        Specifies the method of data corruption, one of the two options: 'Uniform' and 'Binomial'
    
    percentage: float >0 and <1.
        The percentage of non-zero elements to be selected for corruption. 
        
    Returns
    -------
    data_c : numpy ndarray 
        The corrupted data.
    
    x, y, ind : int
        The indices of where corruption is applied. 
    '''
    
    data_c = np.copy(data)
    
    x, y = np.nonzero(data)
    
    ind = np.random.choice(len(x), int(0.1 * len(x)), replace=False)
    
    if method == 'Uniform' else # to be developed
    
    return data_c, x, y, ind