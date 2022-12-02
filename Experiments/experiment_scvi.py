"""
The script to perform the experiments on the Brain Large dataset
@author: HH197
"""

import sys
import os
import time
import anndata
PATH = '/work/long_lab/Hamid' # from the user

sys.path.append(PATH+'/Code')


import torch
import scvi 
import data_prep
from torch.utils.data import random_split

scvi.settings.seed = 197
torch.manual_seed(197)

data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]



brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])


# experiment for scvi 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for j in data_sizes:
    
    loading_batch_size = j if j < 100000 else 100000

    try: 
        os.mkdir(PATH + f'/data_size_{j}')
    except:
        pass
    # adata = scvi.data.brainlarge_dataset(max_cells_to_keep=j, 
    #                                      loading_batch_size=loading_batch_size)
    y_train, _ = train[:j]
    
    adata = anndata.AnnData(y_train.numpy())

    start_time = time.time()
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata)
    model.to_device(device)
    model.train(use_gpu=True)
    model.save(PATH + f"/data_size_{j}/", overwrite=True, save_anndata=True)
    wall_time = time.time() - start_time
    with open(PATH + f"/data_size_{j}/run_time_scvi.txt", "w") as f:
      # Writing time to a file:
      f.write(device.type + f' wall time is: {wall_time}')

    
 
