
"""
Created on Thu Oct 13 10:13:02 2022

@author: HH197
"""

import sys


PATH = '/home/longlab/Desktop' # from the user

sys.path.append(PATH+'/Code')

import torch
import data_prep
from torch.utils.data import random_split

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

torch.manual_seed(197)

data_sizes = [4000, 10000, 15000, 30000]
K = 10

brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])

ro.r('library(zinbwave)')

for j in data_sizes:
    
    y_train, _ = train[:j]
    
    y_train = y_train.numpy()
    rows, cols = y_train.shape
    
    numpy2ri.activate()
    
    X = ro.r.matrix(y_train, nrow=rows, ncol=cols)
    
    ro.r.assign("data", X)
    ro.r.assign("path", PATH + f"/ZINB_org/data_size_{j}/model_zinb_org.rds")
    
    
    ro.r('''
         model_zinb <- readRDS(path)
         mu <- getMu(model_zinb)
         theta <- getTheta(model_zinb)
         logitPi <- getLogitPi(model_zinb)
         loglik <- zinb.loglik(data, mu, theta, logitPi)
         ''')    
    log_like = ro.r("loglik")
    with open(PATH + f"/ZINB_org/data_size_{j}/loglik_ZINB_org.txt", "w") as f:
        f.write(f'log-likelihood train is: {-log_like[0]}, log-likelihood test is:')
 

