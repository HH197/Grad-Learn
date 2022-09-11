"""
The script to perform the experiments on the Brain Large dataset
@author: HH197
"""

import ZINB_grad 
import torch
import data_prep
from torch.utils.data import DataLoader, random_split


PATH = '/home/longlab/Data/Thesis/Data/' # from the user

data_sizes = [4000, 10000, 15000, 30000, 50000, 100000, 1000000]
K = 10
batch_size = 10000


brain = data_prep.Brain_Large(low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])
size = (batch_size, brain.n_select_genes)


for i in data_sizes:
    
    val_size = 10000 if int(i*0.2)>10000 else int(i*0.2)
    
    if i > 15000:
        
        y_val, _ = train[:val_size]
        brain_dl = DataLoader(train,
                              batch_size= batch_size,
                              sampler = data_prep.Indice_Sampler(
                                  torch.arange(val_size, i)),
                              shuffle=False) 
        
        # a random initialziation
        model = ZINB_grad.ZINB_WaVE(Y = torch.randint(0,100, size = size), 
                                    K = K)

        for i, data in enumerate(brain_dl):
            
            batch = data[0].reshape(size) 
            index = data[1]
            
            # Using the alphas, betas, and theta from the previous model.
            model = ZINB_grad.ZINB_WaVE(Y = batch, K = K,
                                        alpha_mu = model.alpha_mu,
                                        alpha_pi = model.alpha_pi,
                                        beta_mu = model.beta_mu,
                                        beta_pi =  model.beta_pi, 
                                       log_theta = model.log_theta)
            
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.08)
            
            losses, neg_log_liks, val_losses = ZINB_grad.train_ZINB_with_val(batch, 
                                                                 y_val, 
                                                                 optimizer, 
                                                                 model)
            
            # save the W and gammas of the trained model
            W_gammas = {key: model.state_dict()[key] for key in ['gamma_mu', 'gamma_pi', 'W']}
            torch.save(W_gammas, PATH + f"W_gammas_iter_{i}.pt")
        
        
        for i, data in enumerate(brain_dl):
            
            batch = data[0].reshape(size) 
            index = data[1]
            
            # Reading the W and Gammas
            W_gammas = torch.load(PATH + f"W_gammas_iter_{i}.pt")
            
            # Using the freezed alphas, betas, and theta from the trained model
            model2 = ZINB_grad.ZINB_WaVE(Y = batch, K = K, 
                                         W = torch.nn.Parameter(W_gammas['W']),
                                         gamma_mu = torch.nn.Parameter(W_gammas['gamma_mu']),
                                         gamma_pi = torch.nn.Parameter(W_gammas['gamma_pi']),
                                        alpha_mu = model.alpha_mu.detach(),
                                        alpha_pi = model.alpha_pi.detach(),
                                        beta_mu = model.beta_mu.detach(),
                                        beta_pi =  model.beta_pi.detach(), 
                                       log_theta = model.log_theta.detach())
            
            optimizer = torch.optim.Adam(model2.parameters(), lr = 0.08)
            losses, neg_log_liks = ZINB_grad.train_ZINB(batch,
                                                        optimizer, 
                                                        model2)
            
            # save the W and gammas of the re-trained model
            W_gammas = {key: model2.state_dict()[key] for key in ['gamma_mu', 'gamma_pi', 'W']}
            torch.save(W_gammas, PATH + f"W_gammas_iter_{i}.pt")

    else: 

        y, _ = train[:i]
        y_train, y_val = random_split(y, [i-val_size, val_size])
        
        model = ZINB_grad.ZINB_WaVE(Y = y_train, K = K)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.08)
        losses, neg_log_liks = ZINB_grad.train_ZINB_early_stopping(y_train, 
                                                                   y_val, 
                                                                   optimizer, 
                                                                   model)