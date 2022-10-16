
"""
Created on Fri Oct 14 11:46:52 2022

@author: HH197
"""

import sys

import anndata
PATH = '/work/long_lab/Hamid' # from the user

sys.path.append(PATH+'/Code')


import torch
import scvi 
import data_prep
from torch.utils.data import random_split, DataLoader
from pyro.distributions import ZeroInflatedNegativeBinomial as ZINB
import numpy as np

scvi.settings.seed = 197
torch.manual_seed(197)

data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]



brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for j in data_sizes:
    
    val_size = np.min([10000, int(j*0.2)])
    y_test, _ = test[10000:(val_size+10000)]
    
    if j <= 100000:
        y_train, _ = train[:j]
        
        vae = scvi.model.SCVI.load(PATH + f'/scvi/data_size_scvi{j}/')
        
        like_params = vae.get_likelihood_parameters()
        p = like_params['mean']/(like_params['mean']+like_params['dispersions']+10**(-4))
        
        
        neg_loglik_train = -ZINB(total_count=torch.tensor(like_params['dispersions'][1]), 
             probs=torch.tensor(p), 
             gate_logits=torch.tensor(like_params['dropout'])).log_prob(y_train).sum()
        
        adata = anndata.AnnData(y_test.numpy())
        like_params_test = vae.get_likelihood_parameters(adata=adata)
        p = like_params_test['mean']/(like_params_test['mean']+like_params_test['dispersions']+10**(-4))
        
        neg_loglik_test = -ZINB(total_count=torch.tensor(like_params_test['dispersions'][1]), 
             probs=torch.tensor(p), 
             gate_logits=torch.tensor(like_params_test['dropout'])).log_prob(y_test).sum()
    else:
        batch_size = 10000
        brain_dl = DataLoader(train,
                              batch_size= batch_size,
                              sampler = data_prep.Indice_Sampler(
                                  torch.arange(j)),
                              shuffle=False)
        neg_loglik_train = []
        for i, data in enumerate(brain_dl):
            
            batch = data[0].reshape((-1, brain.n_select_genes))
            adata = anndata.AnnData(batch.numpy())
            vae = scvi.model.SCVI.load(PATH + f'/scvi/data_size_scvi{j}/', adata=adata)
            
            like_params = vae.get_likelihood_parameters()
            p = like_params['mean']/(like_params['mean']+like_params['dispersions']+10**(-4))
            
            
            neg_loglik_train.append(-ZINB(total_count=torch.tensor(like_params['dispersions'][1]), 
                 probs=torch.tensor(p), 
                 gate_logits=torch.tensor(like_params['dropout'])).log_prob(batch).sum())
        
        neg_loglik_train = sum(neg_loglik_train)
        
        adata = anndata.AnnData(y_test.numpy())
        like_params_test = vae.get_likelihood_parameters(adata=adata)
        p = like_params_test['mean']/(like_params_test['mean']+like_params_test['dispersions']+10**(-4))
        
        neg_loglik_test = -ZINB(total_count=torch.tensor(like_params_test['dispersions'][1]), 
             probs=torch.tensor(p), 
             gate_logits=torch.tensor(like_params_test['dropout'])).log_prob(y_test).sum()
        
    with open(PATH + f"/scvi/data_size_scvi{j}/loglik_scvi.txt", "w") as f:
        f.write(f'log-likelihood train is: {neg_loglik_train}, log-likelihood test is: {neg_loglik_test}')
 
