# Usage 


## Loading data

The `dataset` module contains high-level API to read the data and perform the preprocessing steps. In the following, we load the CORTEX data set into memory and use the Pytorch `DataLoader` module to iterate through the data set with batch size of 128 samples.
```python
from grad.dataset import data_prep
from torch.utils.data import DataLoader

cortex = data_prep.CORTEX()
dl = DataLoader(cortex, batch_size= 128, shuffle=True)
```

## Training a generalized linear model 

The `linear_model` module contains the ZINB-WaVE model with automatic training and validation using gradient descent. In the following: 
1. Loading the CORTEX data set. 
2. Splitting the data into test (20%) and train (80%) sets.
3. Instantiating a ZINB-WaVE model with 10-dimensional latent space.
4. Training with validation and early stopping using GPUs, if available.

```python
from grad.dataset import data_prep
from grad.linear_model import ZINB_grad 
import torch
from torch.utils.data import DataLoader, random_split


device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cortex = data_prep.CORTEX()
test_size = int(cortex.n_cells *0.2)
train, test = random_split(cortex, [cortex.n_cells-test_size, test_size])
y_train, labels_train = train[:]
y_test, labels_test = test[:]

model = ZINB_grad.ZINB_Grad(Y = y_train, K = 10, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.08)
losses, neg_log_liks, val_losses = ZINB_grad.train_ZINB_with_val(y_train,
                                                    y_test, 
                                                    optimizer, 
                                                    model,
                                                    device,
                                                    epochs = 300,
                                                    early_stop = True)
                                                    
```

(to be added: Clustering, visualizing the latent space, data imputation)

