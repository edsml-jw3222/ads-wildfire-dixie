import tensorflow as tf
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Dense

class WildfireLSTM:

    def __init__(self, model_path=None):
        """
        Initializes the WildfireLSTM object. Either loads a pretrained model from a specified path or builds a new LSTM model.

        Parameters
        ----------
        model_path : str, optional
            Path to a pretrained LSTM model (default is None).
        """
        self.model = self.get_pretrained_model(model_path) if model_path else self.build_LSTM_model()


    def build_LSTM_model(self, input_shape=(2, 256, 256, 1)):
        """
        Constructs a new LSTM model with the specified input shape.

        Parameters
        ----------
        input_shape : tuple, optional
            Shape of the input data (default is (None, 256, 256, 1)).

        Returns
        -------
        model : tf.keras.Model
            A compiled Keras model.
        """
        model = Sequential()

        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                            input_shape=input_shape,
                            padding='same', return_sequences=False))  # False because we're predicting the next step only
        model.add(BatchNormalization())

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    

    def train_LSTM_model(self, x_train_fnl, y_train_fnl, batch_size=25, epochs=10, num_fires=125):
        """
        Trains the LSTM model using the provided training datasets.

        Parameters
        ----------
        x_train_datasets : list
            A list of arrays containing input data for training.
            
        y_train_datasets : list
            A list of arrays containing the target data for training.
            
        batch_size : int, optional
            Number of samples per update (default is 25).
            
        epochs : int, optional
            Number of epochs to train the model (default is 10).
            
        num_fires : int, optional
            Number of fire simulations to train the model on (default is 125 based on the Ferguson training data).
        """
        for i in range(num_fires):
            print(f"Training on simulation {i+1} out of {num_fires}")
            x_train = x_train_fnl[i]  
            y_train = y_train_fnl[i]  
            history = self.model.fit(x_train, y_train, batch_size=20, epochs=10)
        

    def get_pretrained_model(self, path):
        """
        Loads an already trained LSTM model from a specified path.

        Parameters
        ----------
        path : str
            Path to the pretrained LSTM model.

        Returns
        -------
        model : tf.keras.Model
            The loaded LSTM model.
        """
        return tf.keras.models.load_model(path)
