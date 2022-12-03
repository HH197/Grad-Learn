
"""
the ZINB-Grad, a gradient-based ZINB GLMM with GPU acceleration,
high-performance scalability, and memory-efficient estimation.
@author: HH197
"""
import torch
from torch import nn
from pyro.distributions import ZeroInflatedNegativeBinomial as ZINB



class ZINB_Grad(nn.Module):
    
    """
    The ZINB-Grad model
    
    A gradient descent-based stochastic optimization process for the ZINB-WaVE
    to overcome the scalability and efficiency challenges inherited in its optimization
    procedure. The result of this combination is ZINB-Grad. 
    
    Parameters
    ----------
    Y : torch.Tensor
        Tensor of shape (n_samples, n_features). 
    device : A `torch.device` object
        Please refer to Pytorch documentation for more details.
    K : int (optional, default=1)
         Number of latent space dimensions
    W : torch.Tensor (optional, default=None)
        Tensor of shape (n_samples, K).
    X : torch.Tensor (optional, default=None)
        Known tensor of shape (n_samples, M). M is user definedand is the number of 
        covariates in X. When X = None, X is a column of ones.
    V : torch.Tensor (optional, default=None)
        Known tensor of shape (n_features, L). L is user defined and is the number of 
        covariates in V. When V = None, V is a column of ones.
    alpha_mu : torch.Tensor (optional, default=None)
        Tensor of shape (K, n_features).
    alpha_pi : torch.Tensor (optional, default=None)
        Tensor of shape (K, n_features).
    beta_mu : torch.Tensor (optional, default=None)
        Tensor of shape (M, n_features).
    beta_pi : torch.Tensor (optional, default=None)
        Tensor of shape (M, n_features).
    gamma_mu : torch.Tensor (optional, default=None)
        Tensor of shape (L, n_samples).
    gamma_pi : torch.Tensor (optional, default=None)
        Tensor of shape (L, n_samples).
    log_theta : torch.Tensor (optional, default=None)
        Tensor of shape (1, n_features). The natural logarithm of the theta parameter in 
        the ZINB distribution.


    O_mu : torch.Tensor (optional, default=None)
        Tensor of shape (n_samples, n_features).
    O_pi : torch.Tensor (optional, default=None)
        Tensor of shape (n_samples, n_features).
    
    Attributes
    ----------
    n : int
        The number of samples
    J : int
        The number of features (genes)
    M : int 
        The number of covariates (columns) in X. 
    
    Examples
    --------
    >>> import ZINB_grad
    >>> import torch
    >>> import data_prep
    >>> cortex = data_prep.CORTEX()
    >>> y, labels = next(iter(DataLoader(cortex, 
                                 batch_size= cortex.n_cells,
                                 shuffle=True)))
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = ZINB_grad.ZINB_WaVE(Y = y, K = 10, device =device)
    """

    def __init__(self,
                 Y,
                 device,
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
        
        self.n, self.J = Y.size() #n samples and J genes
        
        self.X = X
        self.V = V
        self.K = K 
        
        if log_theta == None: 
            self.log_theta = nn.Parameter(torch.rand((1, self.J)))
        else:
            self.log_theta = log_theta
            
          
        if X == None:
            self.X = torch.ones((self.n, 1)).to(device)
            
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
            self.X = X.to(device)
            
            if beta_mu == None:
                self.beta_mu = nn.Parameter(torch.rand(self.M,self.J))
            else: 
                self.beta_mu = beta_mu
                
            if beta_pi == None:    
                self.beta_pi = nn.Parameter(torch.rand(self.M,self.J))
            else:
                self.beta_pi = beta_pi
            
            
        if V == None:
            self.V = torch.ones((self.J, 1)).to(device)
            
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
            self.V = V.to(device)
            
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
        
        """
        The forward method of class Module in torch.nn
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n_samples, n_features).
        
       Returns
       -------
       p : torch.Tensor
            Tensor of shape (n_samples, n_features) which is the probability of failure for each element of 
            data in the ZINB distribution.
        """
        self.log_mu = self.X @ self.beta_mu + self.gamma_mu.T @ self.V.T + \
            self.W @ self.alpha_mu
        self.log_pi = self.X @ self.beta_pi + self.gamma_pi.T @ self.V.T + \
            self.W @ self.alpha_pi
        
        self.mu = torch.exp(self.log_mu)
        self.theta = torch.exp(self.log_theta)
        
        # Adaptive regulatory parameters are applied: 
        p = self.mu/(self.mu + self.theta + 1e-4 + 1e-4*self.mu + 1e-4*self.theta)
        
        return p


    def _loss(self, x, p):
        
        """
        Returns the loss 
        
        A method to calculate the negative log-likelihood, along with the regularization penalty. 
        The regularization is applied to avoid overfitting. 
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n_samples, n_features).
        
       Returns
       -------
       loss : float
            Sum of the negative log-likelihood for all samples. 
       pen : float
            The regularization term loss.
        """
        
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



def train_ZINB(x, optimizer, model, epochs = 150, val = False):
    
    """
    Trains a ZINB-Grad model
    
    The function will train a ZINB-Grad model using an optimizer for a number of epochs, and it 
    will return both losses and negative log-likelihood, which were obtained during the training procedure. 
    
    Parameters
    ----------
    x : torch.Tensor
        It is the data for training, a Tensor of shape (n_samples, n_features). 
    optimizer: An object of torch.optim.Optimizer
        For more details, please refer to Pytorch documentation.
    model: An object of the ZINB_Grad class
        Please refer to the example.
    epochs : int (optional, default=150)
        Number of iteration for training.
    val : bool (optional, default=False)
        Whether it is validation or training process.
    
    Returns
    -------
    losses : list
         A list consisting of the loss of each epoch. 
    neg_log_liks : list
         A list consisting of the negative Log-likelihood of each epoch.
    """
    losses = []
    neg_log_liks = []
    
    for i in range(epochs):

      i += 1 
      batch = x
    
      p = model(batch)
      
      neg_log_lik, pen = model._loss(batch, p)
      loss = neg_log_lik + pen

      losses.append(loss.item())
      neg_log_liks.append(neg_log_lik.item())
      
      if i%50 == 1:
        if not val:
            print(f'epoch: {i:3}  loss: {loss.item():10.2f}') 
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if val:
        print(f'validation loss: {loss.item():10.2f}') 
    else:
        print(f'epoch: {i:3}  loss: {loss.item():10.2f}') # print the last line
    
    
    return losses, neg_log_liks

def train_ZINB_with_val(x,
                        val_data, 
                        optimizer, 
                        model,
                        device,
                        X_val = None,
                        epochs = 150, 
                        PATH = '/home/longlab/Data/Thesis/Data/', 
                        early_stop = False):
    """
    Trains a ZINB-Grad model with validation
    
    The function will train a ZINB-Grad model with validation using an optimizer for a number of epochs, and it 
    will return losses, negative log-likelihood, and validation losses which were obtained during the training procedure. 
    The function will save the model with best validation loss, and it uses early stopping to avoid overfitting. 
    In the early stopping the model with best validation loss will be loaded. 
    
    Parameters
    ----------
    x : `torch.Tensor`
        It is the data for training, a Tensor of shape (n_samples, n_features). 
    val_data : `torch.Tensor`
        It is the validation data, a Tensor of shape (n_samples_val, n_features). 
    optimizer: An object of `torch.optim.Optimizer`
        For more details, please refer to Pytorch documentation.
    model: An object of the `ZINB_Grad` class
        Please refer to the example.
    device : A `torch.device` object
        Please refer to Pytorch documentation for more details.
    X_val : `torch.Tensor` (optional, default=None)
        It is the X parameter of the ZINB-Grad model for the validation samples, a Tensor of shape (n_samples_val, M).
    epochs : int (optional, default=150)
        Number of iteration for training.
    early_stop : bool (optional, default=False)
        If True the function will use early stopping.
    PATH : str
        The path to save the best model.
        
    
    Returns
    -------
    losses : list
         A list consisting of the loss of each epoch. 
    neg_log_liks : list
         A list consisting of the negative Log-likelihood of each epoch.
    val_losses : list
         A list consisting of the validation losses of each validation step.
        
    """
    
    losses = []
    neg_log_liks = []
    val_losses = []
    
    val_loss, _ = val_ZINB(val_data, model, device, X_val = X_val)
    val_losses.append(val_loss)
    
    # to avoid error when training is not making the model any better
    torch.save(model.state_dict(), PATH + 'best_trained_model.pt')
    
    for i in range(epochs):

      i += 1 
      batch = x
    
      p = model(batch)
      
      neg_log_lik, pen = model._loss(batch, p)
      loss = neg_log_lik + pen

      losses.append(loss.item())
      neg_log_liks.append(neg_log_lik.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if i%50 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.2f}')

        val_loss_last, _ = val_ZINB(val_data, model, device, X_val = X_val)
        val_losses.append(val_loss)
        
        if val_loss_last <= val_loss:
            val_loss = val_loss_last
            
            # save model checkpoint
            torch.save(model.state_dict(), PATH + 'best_trained_model.pt')
        elif early_stop:
            model.load_state_dict(torch.load(PATH + 'best_trained_model.pt'))
            break
        
    val_loss_last, _ = val_ZINB(val_data, model, device, X_val = X_val)
    val_losses.append(val_loss)
    
    if val_loss_last <= val_loss:
        val_loss = val_loss_last
        
        # save model checkpoint
        torch.save(model.state_dict(), PATH + 'best_trained_model.pt')
        
    print(f'epoch: {i:3}  loss: {loss.item():10.2f}') # print the last line
    
    return losses, neg_log_liks, val_losses




def val_ZINB(val_data, model, device, epochs = 15, X_val= None): 
    

    """
    Returns the validation loss and negative log-likelihood
    
    The function will perform the validation on a ZINB-Grad model.
    The following parameters would be the same during the validation process: 
        `log_theta`, `beta_mu, `beta_pi`, `alpha_mu`, and `alpha_pi`, and they will 
    not be updated.
    
    However, the `W`, `gamma_mu`, and `gamma_pi` would change because their 
    dimension depend on the number of samples, i.e., are sample specific. 
    
    Parameters
    ----------
    val_data : `torch.Tensor`
        It is the validation data, a Tensor of shape (n_samples_val, n_features). 
    model: An object of the `ZINB_Grad` class
        Please refer to the example.
    device : A `torch.device` object
        Please refer to Pytorch documentation for more details.
    epochs : int (optional, default=15)
        Number of iteration for training.
    X_val : `torch.Tensor` (optional, default=None)
        It is the X parameter of the ZINB-Grad model for the validation samples, a Tensor of shape (n_samples_val, M).

    
    Returns
    -------
    loss : float
         The validation loss 
    neg_log_lik : float
         The validation negative log-likelihood
    """
    

    model_val = ZINB_Grad(Y = val_data,
                          X= X_val,
                          device=device,
                          K = model.K,
                          alpha_mu = model.alpha_mu.detach(),
                          alpha_pi = model.alpha_pi.detach(),
                          beta_mu = model.beta_mu.detach(),
                          beta_pi = model.beta_pi.detach(),
                          log_theta = model.log_theta.detach())
        
    # Tuning the validation model parameters (W and gammas)
    model_val.to(device)
    optimizer = torch.optim.Adam(model_val.parameters(), lr = 0.1)
    losses, neg_log_liks  = train_ZINB(val_data, 
                                       optimizer, 
                                       model_val, 
                                       epochs = epochs, val = True)
    
    return losses[-1], neg_log_liks[-1]


