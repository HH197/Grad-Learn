# Structure
```
├── ZINB_grad.py                : Contains the ZINB_Grad model and high-level training and validation functions. 
├── data_prep.py                : Contains high-level API for data pipelines to load different RNA-seq data sets. 
├── helper.py                   : Contains high-level functions for clustering, visualization, etc.
└── README.md                   : Description of requirements and how to use the ZINB-Grad implementation.
```

# Requirements

ZINB-Grad requires the following frameworks:

- [Pytorch](https://pytorch.org/) 
- [Pyro](http://pyro.ai/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://scipy.org/)
- [h5py](https://www.h5py.org/)
- [Loompy](http://loompy.org/)
- [Matplotlib](https://matplotlib.org/)

# Usage 


## Loading data

The `data_prep` contains high-level API to read the data and perform the preprocessing steps. In the following, we load the CORTEX data set into memory and use the Pytorch `DataLoader` module to iterate through the data set with batch size of 128 samples.
```
import data_prep
from torch.utils.data import DataLoader

cortex = data_prep.CORTEX()
dl = DataLoader(cortex, batch_size= 128, shuffle=True)
```
## Training a ZINB-Grad model 

The `ZINB_grad` contains the ZINB-Grad class, training function, and validation function. In the following: 
1. We load the CORTEX data set into memory. 
2. We split the data into test (20%) and train (80%) sets.
3. We use the `ZINB_grad` module to instantiate and train a ZINB-Grad model with a latent space with 10 dimensions.
4.  We use validation and early stopping for training which will use GPU if it is available.
```
import data_prep
import ZINB_grad 
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

