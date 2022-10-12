"""
The script to perform the experiments on the Brain Large dataset
@author: HH197
"""

import sys
import os
import time

PATH = '/work/long_lab/Hamid' # from the user

sys.path.append(PATH+'/Code')

import ZINB_grad 
import torch
import data_prep
from torch.utils.data import DataLoader, random_split
import numpy as np

# added extra 4000 for warmup
data_sizes = [4000, 4000, 10000, 15000, 30000, 50000, 100000, 1000000]
K = 10
batch_size = 10000


brain = data_prep.Brain_Large(file_dir = PATH + "/Data/1M_neurons_filtered_gene_bc_matrices_h5.h5", 
                              low_memory = False)

test_size = int(brain.n_cells *0.2)
train, test = random_split(brain, [brain.n_cells-test_size, test_size])
size = (batch_size, brain.n_select_genes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for j in data_sizes:
    
    try: 
        os.mkdir(PATH + f'/data_size_{j}')
    except:
        pass
        
    val_size = np.min([10000, int(j*0.2)])

    if j > 15000:
        
        y_val, _ = test[:val_size]
        y_val = y_val.to(device)
        brain_dl = DataLoader(train,
                              batch_size= batch_size,
                              sampler = data_prep.Indice_Sampler(
                                  torch.arange(j)),
                              shuffle=False) 
        start_time = time.time()
        # a random initialziation
        model = ZINB_grad.ZINB_WaVE(Y = torch.randint(0,100, size = size).to(device), 
                                    K = K, device = device)
        
        n_epochs = np.min([round((10**4/j)*300),150])
        
        for i, data in enumerate(brain_dl):
            
            batch = data[0].reshape((-1, brain.n_select_genes)).to(device)
            
            
            # Using the alphas, betas, and theta from the previous model.
            model = ZINB_grad.ZINB_WaVE(Y = batch, K = K, device= device,
                                        alpha_mu = model.alpha_mu,
                                        alpha_pi = model.alpha_pi,
                                        beta_mu = model.beta_mu,
                                        beta_pi =  model.beta_pi, 
                                       log_theta = model.log_theta)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.08)
            
            losses, neg_log_liks, val_losses = ZINB_grad.train_ZINB_with_val(batch, 
                                                                 y_val, 
                                                                 optimizer, 
                                                                 model,
                                                                 device,
                                                                 epochs = n_epochs,
                                                                 PATH = PATH + f'/data_size_{j}/')
            #load the best model in the training process
            model.load_state_dict(torch.load(PATH + \
                                             f'/data_size_{j}/' \
                                                 + 'best_trained_model.pt'))
            
            # save the W and gammas of the trained model
            W_gammas = {key: model.state_dict()[key] for key in ['gamma_mu', 'gamma_pi', 'W']}
            torch.save(W_gammas, PATH + f"/data_size_{j}/W_gammas_iter_{i}.pt")
            torch.save(torch.tensor(losses),PATH +\
                       f"/data_size_{j}/phase_1_losses_iter_{i}.pt")
            torch.save(torch.tensor(val_losses),PATH + \
                       f"/data_size_{j}/phase_1_val_losses_iter_{i}.pt")
        
        #save the best global alpha, beta and theta
        alpha_beta_theta = {key: model.state_dict()[key] for key in ['log_theta', 
                                                                     'beta_mu', 
                                                                     'beta_pi', 
                                                                     'alpha_mu', 
                                                                     'alpha_pi']}
        torch.save(alpha_beta_theta, PATH + f"/data_size_{j}/alpha_beta_theta_best_global.pt")
        
        # new number of iterations for training W and Gammas
        n_epochs = 30
        
        for i, data in enumerate(brain_dl):
            
            batch = data[0].reshape((-1, brain.n_select_genes)).to(device)
            
            # Reading the W and Gammas
            W_gammas = torch.load(PATH + f"/data_size_{j}/W_gammas_iter_{i}.pt")
            
            # Using alphas, betas, and theta from the trained model and freezing them
            model2 = ZINB_grad.ZINB_WaVE(Y = batch, K = K, device = device, 
                                         W = torch.nn.Parameter(W_gammas['W']),
                                         gamma_mu = torch.nn.Parameter(W_gammas['gamma_mu']),
                                         gamma_pi = torch.nn.Parameter(W_gammas['gamma_pi']),
                                        alpha_mu = model.alpha_mu.detach(),
                                        alpha_pi = model.alpha_pi.detach(),
                                        beta_mu = model.beta_mu.detach(),
                                        beta_pi =  model.beta_pi.detach(), 
                                       log_theta = model.log_theta.detach())
            model2.to(device)
            optimizer = torch.optim.Adam(model2.parameters(), lr = 0.08)
            losses, neg_log_liks, val_losses = ZINB_grad.train_ZINB_with_val(batch, 
                                                                 y_val, 
                                                                 optimizer, 
                                                                 model2,
                                                                 device,
                                                                 epochs = n_epochs,
                                                                 PATH = PATH + f'/data_size_{j}/')
            #load the best model in the training process
            model2.load_state_dict(torch.load(PATH + \
                                             f'/data_size_{j}/' \
                                                 + 'best_trained_model.pt'))
            # save the W and gammas of the re-trained model
            W_gammas = {key: model2.state_dict()[key] for key in ['gamma_mu',
                                                                  'gamma_pi',
                                                                  'W']}
            torch.save(W_gammas,PATH + \
                       f"/data_size_{j}/W_gammas_iter_{i}.pt")
            torch.save(torch.tensor(losses),PATH +\
                       f"/data_size_{j}/phase_2_losses_iter_{i}.pt")
            torch.save(torch.tensor(val_losses),PATH + \
                       f"/data_size_{j}/phase_2_val_losses_iter_{i}.pt")
        

    else: 

        y_train, _ = train[:j]
        #y_train, y_val = random_split(y, [j-val_size, val_size])
        #y_train, y_val = y_train[:].to(device), y_val[:].to(device)
        
        y_val, _ = test[:val_size]
        y_train, y_val = y_train.to(device), y_val.to(device)
        
        start_time = time.time()
        model = ZINB_grad.ZINB_WaVE(Y = y_train, K = K, device=device)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.08)
        losses, neg_log_liks, val_losses = ZINB_grad.train_ZINB_with_val(y_train, 
                                                                 y_val, 
                                                                 optimizer, 
                                                                 model,
                                                                 device,
                                                                 PATH = PATH + f'/data_size_{j}/')
        model.load_state_dict(torch.load(PATH + \
                                         f'/data_size_{j}/' \
                                             + 'best_trained_model.pt'))

        torch.save(torch.tensor(losses),PATH + \
                   f"/data_size_{j}/losses.pt")
        torch.save(torch.tensor(val_losses),PATH +\
                   f"/data_size_{j}/val_losses.pt")
        
    wall_time = time.time() - start_time

    with open(PATH + f"/data_size_{j}/run_time_ZINB.txt", "w") as f:
        # Writing time to a file:
        f.write(device.type + f' wall time is: {wall_time}')
 
