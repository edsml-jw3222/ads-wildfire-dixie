import pytest
from wildfire_da import DAwrapper
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(42)
pred_test = np.random.random_sample((5,256,256))
sat_test = np.random.random_sample((5,256,256))

da = DAwrapper(pred_test,sat_test)

def test_compress():
    da.compress(0.95)
    assert(da.predictions_comp.shape == da.satellite_comp.shape)
    assert(da.predictions_comp.shape[1] == da.minPCs)

def test_reconstruct():
    recon = da.reconstruct(da.predictions_comp)
    assert(recon.shape == da.predictions.shape)

def test_assimilate():
    da.assimilate()
    assert(da.predictions_assimilated.shape == da.predictions_comp.shape)