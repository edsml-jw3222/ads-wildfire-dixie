import numpy as np
import pytest

import preprocessing

@pytest.fixture
def sample_data():
    return np.ones((2, 256, 256))

# Test add_channel_axis function
def test_add_channel_axis(sample_data):
    train, test, background = sample_data, sample_data, sample_data
    x_train, x_test, background_lstm = preprocessing.add_channel_axis(train, test, background)
    assert x_train.shape == (2, 256, 256, 1)
    assert x_test.shape == (2, 256, 256, 1)
    assert background_lstm.shape == (2, 256, 256, 1)

def test_reshape_data_for_lstm(sample_data):
    train, test, background = sample_data, sample_data, sample_data
    x_train, y_train, x_test, y_test, background_lstm = preprocessing.reshape_data_for_lstm(train, test, background)
    assert x_train.shape == (2, 256, 256, 1)
    assert y_train.shape == (2, 256, 256, 1)
    assert x_test.shape == (2, 256, 256, 1)
    assert y_test.shape == (2, 256, 256, 1)
    assert background_lstm.shape == (2, 256, 256, 1)

def test_divide_dataset(sample_data):
    datasets = preprocessing.divide_dataset(sample_data, 2)
    assert len(datasets) == 2

def test_reshape_data_for_t10():
    data = np.ones((100, 256, 256))
    inputs, targets = preprocessing.reshape_data_for_t10(data, sequence_length=1, group_size=10)
    assert inputs.shape == (90, 256, 256)
    assert targets.shape == (90, 256, 256)
    
def test_shift_train_data_for_10_steps():
    data = [np.ones((100, 256, 256, 1))]
    x, y = preprocessing.shift_train_data_for_10_steps(data, data)
    assert np.array(x).shape == (1, 80, 2, 256, 256, 1)
    assert np.array(y).shape == (1, 80, 256, 256, 1)

def test_shift_test_data_for_10_steps():
    data = np.ones((30, 256, 256))
    x_test_fnl, y_test_fnl = preprocessing.shift_test_data_for_10_steps(data, data)
    assert x_test_fnl.shape == (10, 2, 256, 256)
    assert y_test_fnl.shape == (10, 256, 256)

def test_shift_background_data():
    data = np.ones((30, 256, 256))
    back_fnl = preprocessing.shift_background_data(data)
    assert back_fnl.shape == (28, 2, 256, 256)