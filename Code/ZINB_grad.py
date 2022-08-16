#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#How to use the optimizer parameters? Should I use them? 
"""
The ZINB-WaVE implementation using the gradient descent approach,
 pytorch, and Pyro.

"""
import torch
from torch import nn
from pyro.distributions import ZeroInflatedNegativeBinomial as ZINB



# The ZINB model 

class ZINB_WaVE(nn.Module):

    def __init__(self,
                 Y,
                 W=None,
                 alpha_mu=None,
                 alpha_pi=None,
                 beta_mu=None,
                 beta_pi=None,
                 gamma_mu=None,
                 gamma_pi=None,
                 log_theta=None,
                 X=None, 
                 V=None, 
                 O_mu=None, 
                 O_pi=None, 
                 K =1):

        super().__init__()
        
        self.out_size = Y.size()
        
        self.n, self.J = self.out_size #n samples and J genes
        
        self.X = X
        self.V = V
        self.K = K 
        
        if log_theta == None: 
            self.log_theta = nn.Parameter(torch.rand((1, self.J)))
        else:
            self.log_theta = log_theta
            
          
        if X == None:
            self.X = torch.ones((self.n, 1))
            
            if beta_mu == None:
                self.beta_mu = nn.Parameter(torch.rand((1, self.J)))
            else: 
                self.beta_mu = beta_mu
                
            if beta_pi == None:    
                self.beta_pi = nn.Parameter(torch.rand((1, self.J)))
            else:
                self.beta_pi = beta_pi
                
        else: 
            _, self.M = X.size()
            
            if beta_mu == None:
                self.beta_mu = nn.Parameter(torch.rand(self.M,self.J))
            else: 
                self.beta_mu = beta_mu
                
            if beta_pi == None:    
                self.beta_pi = nn.Parameter(torch.rand(self.M,self.J))
            else:
                self.beta_pi = beta_pi
            
            
        if V == None:
            self.V = torch.ones((self.J, 1))
            
            if gamma_mu == None:
                self.gamma_mu = nn.Parameter(torch.rand((1, self.n)))
            else: 
                self.gamma_mu = gamma_mu
            
            if gamma_pi == None:  
                self.gamma_pi = nn.Parameter(torch.rand((1, self.n)))
            else: 
                self.gamma_pi = gamma_pi
                
        else: 
            _, self.L = V.size()
            
            if gamma_mu == None:
                self.gamma_mu = nn.Parameter(torch.rand((self.L, self.n)))
            else: 
                self.gamma_mu = gamma_mu
            
            if gamma_pi == None:  
                self.gamma_pi = nn.Parameter(torch.rand((self.L, self.n)))
            else: 
                self.gamma_pi = gamma_pi
            
            
        if W == None:     
            self.W = nn.Parameter(torch.rand((self.n, self.K)))
        else: 
            self.W = W
            
        if alpha_mu == None:
            self.alpha_mu = nn.Parameter(torch.rand((self.K,self.J)))
        else: 
            self.alpha_mu =alpha_mu
        
        if alpha_pi == None:    
            self.alpha_pi = nn.Parameter(torch.rand((self.K,self.J)))
        else: 
            self.alpha_pi = alpha_pi


    def forward(self, x):
        self.log_mu = self.X @ self.beta_mu + self.gamma_mu.T @ self.V.T + \
            self.W @ self.alpha_mu
        self.log_pi = self.X @ self.beta_pi + self.gamma_pi.T @ self.V.T + \
            self.W @ self.alpha_pi
        
        self.mu = torch.exp(self.log_mu)
        self.theta = torch.exp(self.log_theta)
        
        # if you make them double (float64) it will use the memory exponantially 
        p = self.mu/(self.mu + self.theta + 1e-4 + 1e-4*self.mu + 1e-4*self.theta)
        
        return p


    def _loss(self, x, p):
        
        J = x.shape[1]
        n = x.shape[0]
        
        eps_W = J/n
        eps_alpha_mu = 1
        eps_alpha_pi = 1
        eps_theta = J
        
        loss = ZINB(total_count= torch.exp(self.log_theta), 
                    probs = p, 
                    gate_logits = self.log_pi).log_prob(x).sum()
        
        pen = eps_W * torch.linalg.norm(self.W, ord='fro').square().item()/2 + \
          eps_alpha_mu * torch.linalg.norm(self.alpha_mu, ord='fro').square().item()/2 + \
          eps_alpha_pi * torch.linalg.norm(self.alpha_pi, ord='fro').square().item()/2 + \
          eps_theta * torch.var(self.log_theta)/2
        
        return -loss , pen



def train_ZINB(x, optimizer, model, epochs = 300):
    
    losses = []
    neg_log_liks = []
    
    for i in range(epochs):

      i += 1 
      batch = x
    
      p = model(batch)
      
      neg_log_lik, pen = model._loss(batch, p)
      loss = neg_log_lik + pen

      losses.append(loss.detach().numpy())
      neg_log_liks.append(neg_log_lik)
      
      if i%50 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.2f}')
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    print(f'epoch: {i:3}  loss: {loss.item():10.2f}') # print the last line
    
    return losses, neg_log_liks


def val_ZINB(val_data, model, optimizer, epochs = 300): 
    

    """ 
    The parameters in params would be the same during the validation process 
    However, the W, and gammas would change because their dimension depend on 
    the number of samples, which will change with validation and test sets
    
    """

    params = ['log_theta', 'beta_mu', 'beta_pi', 'alpha_mu', 'alpha_pi']
    
    for name, param in model.named_parameters():
        if name in params:
            param.requires_grad = False
    
    model2 = ZINB_WaVE(Y = val_data, K = model.K)
    
    for name in params: 
        setattr(model2, name, getattr(model, name))
    
    losses, neg_log_liks  = train_ZINB(val_data, optimizer, model2, epochs = epochs)
    
    return neg_log_liks[-1]


