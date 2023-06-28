"""
An initial test for testing the packages in production.
"""

import torch
from torch.utils.data import DataLoader

import grad.ZINB_grad as ZINB_grad


def test_learning(synth_data, device):

    torch.manual_seed(197)


    y = next(iter(DataLoader(
        synth_data,
        batch_size=synth_data.n_samples,
        shuffle=True
    )))

    model = ZINB_grad.ZINB_Grad(Y=y, K=10, device=device)
    # y = y.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    _, _ = ZINB_grad.train_ZINB(y, optimizer, model, epochs=150)

    p = model(y)
    assert not torch.any(torch.isclose(p, synth_data.p))

# helper.plot_line(list(range(len(losses))), losses)
#
# w = model.W.cpu().detach().numpy()
#
# sil_coeff = helper.kmeans(w)
#
# helper.plot_line(list(range(len(sil_coeff))), sil_coeff)
#
#
# labels = labels.numpy()
#
# helper.measure_q(w, labels, n_clusters=7)
