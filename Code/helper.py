
"""
This contains all of the helper functions for experiments and visualizations.
@author: HH197
"""

import numpy as np
# import seaborn as sn
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
# from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering


def kmeans(data,
           range_cluster = (2, 10),
           kmeans_kwargs = {"init": "random", 
                                   "n_init": 50, 
                                   "max_iter": 400, 
                                   "random_state": 197}):
        
    """
    Performs K-means on the data
    
    Perfoms K-means on the data (usually latent space of a model) for various number of clusters 
    and returns a list containing average Silhouette width. 
    
    Parameters
    ----------
    kmeans_kwargs : dict
        For more details, please refer to scikit-learn documentation.
    range_cluster : tuple
        The range of the number of clusters.

    """
    
    sil_coeff = []
    
    for k in range(range_cluster[0], range_cluster[1]):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        # score = kmeans.inertia_
        score = metrics.silhouette_score(data, kmeans.labels_)
        sil_coeff.append(score)
        
    return sil_coeff

def plot_line (x, y, 
               line_style = None,
               axis_x_ticks = None,
               color = 'blue', 
               xlab = "Number of epochs", 
               ylab = 'Neg-Loglikelihood'):
    '''
    A simple function to graph the plot lines with customized options.
  
    Parameters
    ----------
    y : list 
        The y coordinate of the points.
    x : list
        The x coordinate of the points.
    xlab : str
        The x axis label.
    ylab : str
        The y axis label.
    '''

    fig, ax = plt.subplots()
      
    if line_style == None: 
      ax.plot(x, y, color = color)
    else: 
      ax.plot(x, y, line_style, color = color)
      
    if axis_x_ticks != None: 
      ax.set_xticks(axis_x_ticks)
      ax.set_xticklabels(axis_x_ticks)
      
    ax.spines["top"].set_color(None)
    ax.spines["right"].set_color(None)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.grid(None)
    plt.show()



def measure_q(data, Groups= None, n_clusters=6,  
              
              kmeans_kwargs = {"init": "random", 
                               "n_init": 50, 
                               "max_iter": 400, 
                               "random_state": 197}):

    '''
    This function will measure the quality of clustering using NMI, ARI, and ASW. 
    
    Parameters
    ----------
    data : an array 
        The latent space of the model.
    
    Groups : an array
        The real lables.
    
    n_clusters : int
        The number of clusters in K-means.
    '''
    Groups = np.ndarray.astype(Groups, np.int)
            
    kmeans = KMeans(n_clusters, **kmeans_kwargs)
    kmeans.fit(data)
    
    NMI = metrics.cluster.normalized_mutual_info_score(Groups, kmeans.labels_)
    print(f'The NMI score is: {NMI}')
    
    ARI = metrics.adjusted_rand_score(Groups, kmeans.labels_)
    print(f'The ARI score is: {ARI}')
    
    ASW = metrics.silhouette_score(data, kmeans.labels_)
    print(f'The ASW score is: {ASW}')
    
    # CM = metrics.confusion_matrix(Groups, kmeans.labels_)
    
    # df_cm = DataFrame(CM, range(1, CM.shape[0]+1), range(1, CM.shape[1]+1))
    # # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 5}) # font size
    
    # plt.show()

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
        Specifies the method of data corruption, one of the two options: "Uniform" and "Binomial"
    
    percentage: float >0 and <1.
        The percentage of non-zero elements to be selected for corruption. 
        
    Returns
    -------
    data_c : numpy ndarray 
        The corrupted data.
    
    x, y, ind : int
        The indices of where corruption is applied. 
    '''
    
    data_c = data.astype(np.int32)
    x, y = np.nonzero(data)
    ind = np.random.choice(len(x), int(0.1 * len(x)), replace=False)
    
    if method == 'Uniform':
        data_c[x[ind], y[ind]] *= np.random.binomial(1, p)
        
    elif method == 'Binomial':
        
        data_c[x[ind], y[ind]] = np.random.binomial(data_c[x[ind], y[ind]].astype(np.int), p)
        
    else:
        raise ValueError('''Method can be one of "Uniform" or "Binomial"''') 
    # to be developed
    
    return data_c.astype(np.float32), x, y, ind

def Eval_Imputation (data, data_imp, x, y, ind):
    
    '''
    Calculates the median L1 distance between the original dataset and the 
    imputed values for corrupted entries only.
    
    Parameters
    ----------
    data : numpy ndarray 
        The data.
        
    data_imp : numpy ndarray 
        The imputed data.
        
    x, y, ind : int
        The indices of where corruption is applied. 
        
    Returns
    -------
    L1 : float
        The median L1 distance between original and imputed datasets at given
    indices.
    '''
    
    L1 = np.median(np.abs(data[x[ind], y[ind]] - data_imp[x[ind], y[ind]]))
    
    return L1



def entropy(batches):
    
    '''
    Calculates the entropy.
    
    Entropy of mixing for c different batches is defined as: 
        
        $$E = - \sum_{i=1}^c x_i \log x_i$$
    
    where $x_i$ is the proportion of cells from batch i in a given region, such
    that $\sum_{i=1}^c x_i = 1$. 
    
    Parameters
    ----------
    batches : numpy array or list
        The batches in the region.
    
    Returns
    -------
    entropy : float
        Entropy of mixing.
    
    '''    
    n_batches, frq = np.unique(batches, return_counts=True)
    n_batches = len(n_batches)
    frq = frq/np.sum(frq)
    
    return -np.sum(frq*np.log(frq))



def entropy_batch_mixing(latent_space, 
                         batches, 
                         K = 50, 
                         n_jobs = 8, 
                         n = 100, 
                         n_iter = 50):
    
    '''
    Adopted from:
    
    1) Haghverdi L, Lun ATL, Morgan MD, Marioni JC. Batch effects in
    single-cell RNA-sequencing data are corrected by matching mutual nearest 
    neighbors. Nat Biotechnol. 2018 Jun;36(5):421-427. doi: 10.1038/nbt.4091. 
    
    2) Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative 
    modeling for single-cell transcriptomics. Nat Methods. 
    2018 Dec;15(12):1053-1058. doi: 10.1038/s41592-018-0229-2. 
    
    This function will choose `n` cells from batches, finds `K` nearest neighbors
    of each randomly chosen cell, and calculates the average regional entropy
    of all `n` cells. 
    
    The procedure is repeated for `n_iter` iterations. Finally, the average of the 
    iterations is returned as the final batch mixing score. 
    
    Parameters
    ----------
    latent_space : numpy ndarray 
        The latent space matrix.
        
    batches : a numpy array or a list
        The batch number of each sample in the latent space matrix.
        
    K : int
        Number of nearest neighbors. 
    
    n_jobs : int
        Number of jobs. Please visit scikit-learn documentation for more info. 
    
    n : int
        Number of cells to be chosen randomly. 
    
    n_iter : int
        Number of iterations to randomly choosing n cells.
    
    Returns
    -------
    score : float <= 1 
        The batch mixing score; the higher, the better.
    
    '''
    
    n_samples = latent_space.shape[0]
    nne = NearestNeighbors(n_neighbors=K+1, n_jobs=n_jobs)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(n_samples)
    
    
    score = 0
    
    for t in range(n_iter):
        ind = np.random.choice(n_samples, size=n)
        inds = kmatrix[ind].nonzero()[1].reshape(n, K)
        score += np.mean([entropy(batches[inds[i]])\
                      for i in range(n)])

    score = score/n_iter
    
    return score


def plot_tSNE(latent, 
              labels, 
              cmap = plt.get_cmap("tab10", 7), 
              perform_tsne = True):
    
    '''
    Adoptd from:
    
    Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative 
    modeling for single-cell transcriptomics. Nat Methods. 
    2018 Dec;15(12):1053-1058. doi: 10.1038/s41592-018-0229-2. 
    
    Given a `latent` space and class labels, the function will calculate the 
    tSNE of the latent space and make a graph of the tSNE latent space using
    the classes.
    
    
    Parameters
    ----------
    latent_space : numpy ndarray 
        The latent space matrix.
        
    labels : a numpy array or a list
        The batch (or cluster) number of each sample in the latent space matrix.
        
    cmap : pylot instance
        a colormap instance (see Matplotlib doc for more info).
    
    perform_tsne : Boolean
        If `True` the function will perform the tSNE. Otherwise, tSNE will not
        be performed on the latent space.
    
    
    Returns
    -------
    latent : numpy ndarray
        The latent space of the tSNE.
    
    '''
    
    if perform_tsne:
        latent = TSNE().fit_transform(latent)
    
    
    plt.figure(figsize=(10, 10))
    plt.scatter(latent[:, 0], latent[:, 1], c=labels, \
                                   cmap=cmap, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    
    return latent
      

def HC(latent, 
        labels, 
        num_clusters = [2, 3, 4, 7]):
    
    '''
    Given a `latent` space and class labels, the function will perform hierarchical 
    clustering (HC) on the latent space and calculate the NMI score.
    
    
    Parameters
    ----------
    latent_space : numpy ndarray 
        The latent space matrix.
        
    labels : a numpy array or a list
        The batch (or cluster) number of each sample in the latent space matrix.
        
    num_clusters : list
        A list of the number of cluusters to perform the HC.

    
    Returns
    -------
    nmis : list
        The NMI score of each number of clusters.
    
    '''
    
    nmis = []
    
    for i in num_clusters: 

        new_labels = labels if i == 7 else np.zeros_like(labels)
        
        if i == 2:
            for t in np.arange(labels.shape[0]):
                new_labels[t] = 1 if labels[t] in [2, 6, 5] else 0
        
        if i == 3:
            for t in np.arange(labels.shape[0]):
                if labels[t] in [2, 6, 5]:
                    new_labels[t] = 2
                elif labels[t] in [1, 3]: 
                    new_labels[t] = 1
                else:
                    new_labels[t] = 0
                    
        if i == 4:
            for t in np.arange(labels.shape[0]):
                if labels[t] in [2, 6, 5]:
                    new_labels[t] = 3
                elif labels[t] in [1, 3]: 
                    new_labels[t] = 2
                elif labels[t] in [0]:
                    new_labels[t] = 1
                else:
                    new_labels[t] = 0
        
        
            
        hc = AgglomerativeClustering(n_clusters = i).fit(latent)
        nmi = metrics.cluster.normalized_mutual_info_score(new_labels,
                                                     hc.labels_)
        nmis.append(nmi)
        
    return nmis
