import numpy as np
import tensorflow as tf
from keras import layers, Model, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

class WildfireVAE:
    def __init__(self, input_shape=(256, 256, 1), latent_dim=128, reg_rate=0.01):
        """
        Initialize the WildfireVAE class.

        Args:
        input_shape (tuple): The shape of the input data. Default is (256, 256, 1).
        latent_dim (int): The dimensionality of the latent space. Default is 64.
        reg_rate (float): The regularization rate. Default is 0.001.
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.reg_rate = reg_rate
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_vae()

    def sampling(self, args):
        """
        Reparameterization by sampling from a Gaussian.

        Args:
        args (tensor): mean and log of variance of Q(z|X)

        Returns:
        z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.keras.backend.exp(z_log_var / 2) * epsilon

    def build_encoder(self):
        """
        Build the encoder part of the VAE.

        Returns:
        model: A Model that includes the encoder part of the VAE.
        """
        encoder_inputs = tf.keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(128, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        z_mean = layers.Dense(self.latent_dim, kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        z_log_var = layers.Dense(self.latent_dim, kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        z = layers.Lambda(self.sampling)([z_mean, z_log_var])
        return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


    def build_decoder(self):
        """
        Build the decoder part of the VAE.

        Returns:
        model: A Model that includes the decoder part of the VAE.
        """
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 32 * 128, kernel_regularizer=regularizers.l2(self.reg_rate))(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Reshape((32, 32, 128))(x)
        x = layers.Conv2DTranspose(64, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(32, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(32, 2, strides=2, padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 2, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(self.reg_rate))(x)

        return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")


    def build_vae(self, learning_rate=0.0001):
        """
        Build the complete VAE by connecting the encoder and decoder.

        Returns:
        model: A Model that includes the complete VAE.
        """
        encoder_inputs = tf.keras.Input(shape=self.input_shape)
        z_mean, z_log_var, z = self.encoder(encoder_inputs)
        vae_outputs = self.decoder(z)
        vae = tf.keras.Model(encoder_inputs, vae_outputs, name="vae")

        reconstruction_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, vae_outputs)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=(1, 2))  # Calculate the mean over the image dimensions
        reconstruction_loss *= self.input_shape[0] * self.input_shape[1]
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        # Compile
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.7))

        return vae

  
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=30):
        """
        Train the VAE model with the provided data.

        Args:
        x_train (numpy array): Training data.
        x_test (numpy array): Validation data
        batch_size (int): Number of samples per gradient update. Default is 32.
        epochs (int): Number of epochs to train the model. Default is 100.
        """
        # self.model.fit(x, epochs=epochs, batch_size=batch_size)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        ]

        history = self.model.fit(x_train, y_train, epochs=30, batch_size=45, callbacks=callbacks, validation_data=(x_test, y_test))        
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Loss', 'Val_loss'])

    def load_pretrained_model(self, path):
        """
        Load a pretrained VAE model

        Args:
        path (string): the path to the tensorflow model
        """
        self.model = tf.keras.models.load(path)       

    def predict(self, test):
        """
        Generate wildfire predictions for time step t+1

        Args:
        test (numpy array): the test data
        """
        pred = self.model.predict(test)
        return pred
