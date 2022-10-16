
"""
Created on Thu Oct 13 10:13:02 2022

@author: HH197
"""

import sys
import os
import time

PATH = '/home/longlab/Data/Thesis' # from the user

sys.path.append(PATH+'/Code')

import torch
import data_prep
from torch.utils.data import random_split

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

torch.manual_seed(197)

data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]
K = 10

brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])

ro.r('library(zinbwave)')

for j in data_sizes:
    print(j)
    try: 
        os.mkdir(PATH + f'/data_size_{j}')
    except:
        pass

    y_train, _ = train[:j]
    
    y_train = y_train.numpy()
    rows, cols = y_train.shape
    
    numpy2ri.activate()
    
    X = ro.r.matrix(y_train, nrow=rows, ncol=cols)
    
    ro.r.assign("data", X)
    ro.r.assign("path", PATH + f"/data_size_{j}/model_zinb_org.rds")
    
    start_time = time.time()
    
    ro.r('''
    K = 10
    model_zinb <- zinbModel(n=NROW(data), J=NCOL(data), K=K)
    model_zinb <- zinbInitialize(model_zinb, data)
    model_zinb <- zinbOptimize(model_zinb, data)
    ''')
    
    ro.r('''
         saveRDS(model, file = path)
         ''')    
    
    wall_time = time.time() - start_time

    with open(PATH + f"/data_size_{j}/run_time_ZINB_org.txt", "w") as f:
        # Writing time to a file:
        f.write(f' wall time is: {wall_time}')
 

