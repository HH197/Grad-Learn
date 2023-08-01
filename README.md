![Tests](https://github.com/HH197/Grad-Learn/actions/workflows/test.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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

## Install Grad-Learn from source:

```sh
git clone https://github.com/HH197/ZINB-Grad
cd ZINB-Grad
pip install .
```

## Install Grad-Learn in development mode:

Here’s how to set up Grad-Learn for local development:

```sh
git clone https://github.com/HH197/ZINB-Grad
cd ZINB-Grad
pip pip install -e ".[dev,docs]"
```

## Supported Operating Systems

Grad-Learn is compatible with the following operating systems:

- Linux (tested on Ubuntu)

Please note that while Grad-Learn is expected to work on other Linux distributions, we provide official support for 
Ubuntu.

## Citation

If you use Grad-learn in a scientific publication, we would appreciate citations to the following paper:

[Gradient-based implementation of linear model outperforms deep learning models](https://www.biorxiv.org/content/10.1101/2023.07.29.551062v1)

Bibtex entry:
```
{Grad-Learn,
	author = {Hamid Hamidi and Dinghao Wang and Quan Long and Matthew Greenberg},
	title = {Gradient-based implementation of linear model outperforms deep learning models},
	elocation-id = {2023.07.29.551062},
	year = {2023},
	doi = {10.1101/2023.07.29.551062},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning has been widely considered more effective than traditional statistical models in modeling biological complex data such as single-cell omics. Here we show the devil is hidden in details: by adapting a modern gradient solver to a traditional linear mixed model, we showed that conventional models can outperform deep models in terms of both speed and accuracy. This work reveals the potential of re-implementing traditional models with modern solvers.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/07/29/2023.07.29.551062},
	eprint = {https://www.biorxiv.org/content/early/2023/07/29/2023.07.29.551062.full.pdf},
	journal = {bioRxiv}
}
```
