import pytest
import torch

from grad import data_prep


@pytest.fixture(scope="session")
def device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# Adopted from scvi
@pytest.fixture(scope="session")
def synth_data(device):
    """Docstring for model_fit."""
    return data_prep.SyntheticData(device)
