"""
The script to perform the experiments on the Brain Large dataset
@author: HH197
"""

import sys
import os
import time

PATH = '/work/long_lab/Hamid' # from the user

sys.path.append(PATH+'/Code')


import torch
import scvi 

data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]

# experiment for scvi 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for j in data_sizes:
  loading_batch_size = j if j < 100000 else 100000
  print(j)

  adata = scvi.data.brainlarge_dataset(max_cells_to_keep=j, 
                                       loading_batch_size=loading_batch_size)
  start_time = time.time()
  scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
  model = scvi.model.SCVI(adata)
  model.to_device(device)
  model.train(use_gpu=True)
  wall_time = time.time() - start_time
  with open(PATH + f"/data_size_{j}/run_time_scvi.txt", "w") as f:
    # Writing time to a file:
    f.write(device.type + f' wall time is: {wall_time}')

    
 
