
"""
Created on Fri Oct 14 12:54:41 2022

@author: HH197
"""

import sys

PATH = '/work/long_lab/Hamid' # from the user

sys.path.append(PATH+'/Code')

import ZINB_grad 
import torch
import data_prep
from torch.utils.data import random_split, DataLoader
import numpy as np

# added extra 4000 for warmup
data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]
K = 10
batch_size = 10000
torch.manual_seed(197)


brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])
size = (batch_size, brain.n_select_genes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for j in data_sizes:
    print(j)
    
    val_size = np.min([10000, int(j*0.2)])
    y_test, _ = test[10000:(val_size+10000)]
    y_test = y_test.to(device)
    
    if j > 15000:
        brain_dl = DataLoader(train,
                              batch_size= batch_size,
                              sampler = data_prep.Indice_Sampler(
                                  torch.arange(j)),
                              shuffle=False) 
        model = ZINB_grad.ZINB_WaVE(Y = torch.randint(0,100, size = size).to(device), 
                                    K = K, device = device)
        
        alpha_beta_theta = torch.load(PATH + f"/Oct12_ZINB_Grad/data_size_{j}/alpha_beta_theta_best_global.pt",
                                       map_location=device)
        neg_loglik_train = []
        for i, data in enumerate(brain_dl):
            batch = data[0].reshape((-1, brain.n_select_genes)).to(device)
            
            W_gammas = torch.load(PATH + f"/Oct12_ZINB_Grad/data_size_{j}/W_gammas_iter_{i}.pt",
                                   map_location=device)
            model = ZINB_grad.ZINB_WaVE(Y = batch, K = K, device = device, 
                                         W = W_gammas['W'],
                                         gamma_mu = W_gammas['gamma_mu'],
                                         gamma_pi = W_gammas['gamma_pi'],
                                        alpha_mu = alpha_beta_theta['alpha_mu'],
                                        alpha_pi = alpha_beta_theta['alpha_pi'],
                                        beta_mu = alpha_beta_theta['beta_mu'],
                                        beta_pi =  alpha_beta_theta['beta_pi'], 
                                       log_theta = alpha_beta_theta['log_theta'])
            
            with torch.no_grad():
                p = model(batch)
                neg_loglik, _ = model._loss(batch, p)
                neg_loglik = neg_loglik.item()
                neg_loglik_train.append(neg_loglik)
        
        model.to(device)
        neg_loglik_train = sum(neg_loglik_train)
        _, neg_loglik_test = ZINB_grad.val_ZINB(y_test, model, device, 30)

    else:

        y_train, _ = train[:j]
        y_train = y_train.to(device)
        
        model = ZINB_grad.ZINB_WaVE(Y = y_train, K = K, device=device)
        args = torch.load(PATH + f'/Oct12_ZINB_Grad/data_size_{j}/best_trained_model.pt', 
                          map_location=device)
        model.load_state_dict(args)
        model.to(device)
        with torch.no_grad():
            p = model(y_train)
            neg_loglik_train, _ = model._loss(y_train, p)
            neg_loglik_train = neg_loglik_train.item()
        
        _, neg_loglik_test = ZINB_grad.val_ZINB(y_test, model, device, 30)

    with open(PATH + f"/Oct12_ZINB_Grad/data_size_{j}/loglik_ZINB_Grad.txt", "w") as f:
        f.write(f'log-likelihood train is: {neg_loglik_train}, log-likelihood test is: {neg_loglik_test}')
 
        
        

