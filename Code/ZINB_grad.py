#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The ZINB-WaVE implementation using the gradient descent approach,
 pytorch, and Pyro.

"""
import torch
from torch import nn
from pyro.distributions import ZeroInflatedNegativeBinomial



# The ZINB model 

class ZINB_WaVE(nn.Module):

  def __init__(self,
               Y, 
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

    self.log_theta = nn.Parameter(torch.rand((1, self.J)))
      
    if X == None:
      self.X = torch.ones((self.n, 1))
      self.beta_mu = nn.Parameter(torch.rand((1, self.J)))
      self.beta_pi = nn.Parameter(torch.rand((1, self.J)))
    else: 
      _, self.M = X.size()
      self.beta_mu = nn.Parameter(torch.rand(self.M,self.J))
      self.beta_pi = nn.Parameter(torch.rand(self.M,self.J))

    if V == None:
      self.V = torch.ones((self.J, 1))
      self.gamma_mu = nn.Parameter(torch.rand((1, self.n)))
      self.gamma_pi = nn.Parameter(torch.rand((1, self.n)))
    else: 
      _, self.L = V.size()
      self.gamma_mu = nn.Parameter(torch.rand((self.L, self.n)))
      self.gamma_pi = nn.Parameter(torch.rand((self.L, self.n)))

    self.W = nn.Parameter(torch.rand((self.n, self.K)))
    self.alpha_mu = nn.Parameter(torch.rand((self.K,self.J)))
    self.alpha_pi = nn.Parameter(torch.rand((self.K,self.J)))


  def forward(self, x):
    self.log_mu = self.X @ self.beta_mu + self.gamma_mu.T @ self.V.T + self.W @ self.alpha_mu
    self.log_pi = self.X @ self.beta_pi + self.gamma_pi.T @ self.V.T + self.W @ self.alpha_pi
    
    self.mu = torch.exp(self.log_mu)
    self.theta = torch.exp(self.log_theta)

    p = self.mu/(self.mu + self.theta + 1e-4)

    return p




def loss_zinb(x, p, model):
    
    J = x.shape[1]
    n = x.shape[0]
    
    eps_W = J/n
    eps_alpha_mu = 1
    eps_alpha_pi = 1
    eps_theta = J
    
    loss = ZeroInflatedNegativeBinomial(total_count= torch.exp(model.log_theta), 
                                                  probs = p, 
                                                  gate_logits = model.log_pi).log_prob(x).sum()
    
    pen = eps_W * torch.linalg.norm(model.W, ord='fro').square().item()/2 + \
      eps_alpha_mu * torch.linalg.norm(model.alpha_mu, ord='fro').square().item()/2 + \
      eps_alpha_pi * torch.linalg.norm(model.alpha_pi, ord='fro').square().item()/2 + \
      eps_theta * torch.var(model.log_theta)/2
    
    return -loss , pen



def train_ZINB(optimizer, x, model, epochs = 300):
    
    losses = []
    neg_log_liks = []
    for i in range(epochs):

      i += 1 
      batch = x
    
      p = model(batch)
      
      neg_log_lik, pen = loss_zinb(batch, p, model)
      loss = neg_log_lik + pen

      losses.append(loss.detach().numpy())
      neg_log_liks.append(neg_log_lik)
      
      if i%50 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
    
    return losses, neg_log_liks


y1 = torch.from_numpy(y)

model = ZINB_WaVE(Y = y1, K = 10)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

losses, neg_log_liks = train_ZINB(optimizer, y1, model, epochs = 300)


