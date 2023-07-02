# Grad-Learn

Grad-Learn is written in Python and supported by PyTorch and Pyro on the backend.
Grad-Learn enables flexible and scalable statistical and machine learning, unifying the best of modern deep learning and 
classic machine learning.

With the ever-growing size of the data sets, the demand for highly-efficient, scalable machine learning solutions is 
increasing. Usually, it is considered that deep learning models outperform other techniques if the data size is large. 
However, if the underlying structure of the data does not require the amount of non-linearity introduced by deep models,
this might not be the case. 

In this scenario, one would choose traditional machine learning or statistical learning models since they are more 
interpretable. However, the capabilities of these methods are limited by computing time and complexity for large-scale 
problems. For this reason and to bridge the gap between traditional machine learning models and large-scale data sets, 
we developed the Grad-Learn.

## Structure
```
../grad
├── dataset                     : Contains high-level API for loading different data sets.
├── linear_model                : Contains linear models, including linear regression, GLMs, and GLMMs.
└── utils                       : Contains high-level functions for clustering, visualization, etc.
```

**Install Grad-Learn from source:**

```sh
git clone https://github.com/HH197/ZINB-Grad
cd ZINB-Grad
pip install .
```

**Install Grad-Learn in development mode:**

Here’s how to set up Grad-Learn for local development:

```sh
git clone https://github.com/HH197/ZINB-Grad
cd ZINB-Grad
pip pip install -e ".[dev,docs]"
```


