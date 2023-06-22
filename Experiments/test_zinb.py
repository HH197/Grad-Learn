"""
A Test for testing the packages in production.
Author: HH197
"""

import sys

sys.path.append("/home/hh197/Data/Projects/ZINB-Grad/src/")

import grad.ZINB_grad as ZINB_grad
import torch
import numpy as np
import grad.data_prep as data_prep
import grad.helper as helper
from importlib import reload
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
import pandas as pd


torch.manual_seed(197)
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cortex = data_prep.CORTEX(file_dir="/home/hh197/Data/Thesis/Data/new.txt")
y, labels = next(iter(DataLoader(cortex, batch_size=cortex.n_cells, shuffle=True)))


model = ZINB_grad.ZINB_Grad(Y=y, K=10, device=device)
y = y.to(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
losses, neg_log_liks = ZINB_grad.train_ZINB(y, optimizer, model, epochs=150)

helper.plot_line(list(range(len(losses))), losses)

w = model.W.cpu().detach().numpy()

sil_coeff = helper.kmeans(w)

helper.plot_line(list(range(len(sil_coeff))), sil_coeff)


labels = labels.numpy()

helper.measure_q(w, labels, n_clusters=7)
