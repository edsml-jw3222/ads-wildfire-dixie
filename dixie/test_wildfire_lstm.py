import pytest
import tensorflow as tf
from wildfire_lstm import WildfireLSTM  # Replace 'your_module' with the actual module where WildfireLSTM is defined


def test_init_no_model_path():
    model = WildfireLSTM()
    assert isinstance(model.model, tf.keras.Model)


def test_build_LSTM_model():
    model = WildfireLSTM()
    new_model = model.build_LSTM_model()
    assert isinstance(new_model, tf.keras.Model)
    assert len(new_model.layers) == 3

def test_train_LSTM_model():
    model = WildfireLSTM()
    # Replace with the actual data
    x_train_fnl = [tf.random.normal(shape=(1, 2, 256, 256, 1)) for _ in range(2)]
    y_train_fnl = [tf.random.normal(shape=(1, 256, 256, 1)) for _ in range(2)]
    model.train_LSTM_model(x_train_fnl, y_train_fnl, num_fires=2)
