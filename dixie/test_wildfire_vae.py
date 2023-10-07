import pytest
import numpy as np
from wildfire_vae import WildfireVAE
import tensorflow as tf

class TestWildfireVAE:
    def setup_class(self):
        self.input_shape = (256, 256, 1)
        self.latent_dim = 64
        self.reg_rate = 0.001
        self.vae = WildfireVAE(input_shape=self.input_shape, 
                               latent_dim=self.latent_dim, 
                               reg_rate=self.reg_rate)

    
    def test_build_encoder(self):
        encoder = self.vae.build_encoder()
        assert encoder.input_shape[1:] == self.input_shape
        assert isinstance(encoder.output_shape, list)
        assert len(encoder.output_shape) == 3
        assert encoder.output_shape[0][1:] == (self.latent_dim,)
        assert encoder.output_shape[1][1:] == (self.latent_dim,)
        assert encoder.output_shape[2][1:] == (self.latent_dim,)

    def test_build_decoder(self):
        decoder = self.vae.build_decoder()
        assert decoder.input_shape[1:] == (self.latent_dim,)
        assert decoder.output_shape[1:] == self.input_shape

    def test_build_vae(self):
        vae = self.vae.build_vae()
        assert vae.input_shape[1:] == self.input_shape
        assert vae.output_shape[1:] == self.input_shape

    def test_train(self):
        x = np.random.rand(10, *self.input_shape)
        self.vae.train(x, x, x, x, batch_size=2, epochs=1)
