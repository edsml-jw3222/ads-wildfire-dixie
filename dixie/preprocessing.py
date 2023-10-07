import numpy as np
import os
def load_data(path):
    """
    Load train, test, background and observation data from the given .npy files.

    Returns
    -------
    tuple
        Train data, Test data, Background data, Observation data.
    """
    train = np.load(os.path.join(path,'Ferguson_fire_train.npy'))
    test = np.load(os.path.join(path,'Ferguson_fire_test.npy'))
    background = np.load(os.path.join(path,'Ferguson_fire_background.npy'))
    obs = np.load(os.path.join(path,'Ferguson_fire_obs.npy'))
    return train, test, background, obs

def add_channel_axis(train, test, background):
    """
    Add an extra dimension for channels to the input datasets.

    Parameters
    ----------
    train : ndarray
        Training dataset.
    test : ndarray
        Test dataset.
    background : ndarray
        Background data.

    Returns
    -------
    tuple
        Training dataset, Test dataset and Background data with an added dimension.
    """
    x_train = train[..., np.newaxis]
    x_test = test[..., np.newaxis]
    background_lstm = background[..., np.newaxis]
    return x_train, x_test, background_lstm


def reshape_data_for_lstm(train, test, background):
    """
    Reshape input data for LSTM. Add an extra dimension and reshape the targets.

    Parameters
    ----------
    train : ndarray
        Training dataset.
    test : ndarray
        Test dataset.
    background : ndarray
        Background data.

    Returns
    -------
    tuple
        Reshaped Training and Test datasets, and Background data.
    """
    rows,cols=np.shape(train)[1],np.shape(train)[2]

    x_train, x_test, background_lstm = add_channel_axis(train, test, background)

    y_train = train.reshape((train.shape[0], rows, cols, 1))

    y_test = test.reshape((test.shape[0], rows, cols, 1))

    print("Shape of x_train: ", x_train.shape) 
    print("Shape of y_train: ", y_train.shape)
    print("Shape of x_test: ", x_test.shape) 
    print("Shape of y_test: ", y_test.shape)
    print("Shape of background: ", background_lstm.shape)

    return x_train, y_train, x_test, y_test, background_lstm


def reshape_data_for_vae(train, test):
    """
    Reshape input data for Variational Autoencoder (VAE).

    Parameters
    ----------
    train : ndarray
        Training dataset.
    test : ndarray
        Test dataset.

    Returns
    -------
    tuple
        Reshaped Training and Test datasets.
    """
    x_train, y_train = reshape_data_for_t10(train)
    x_test, y_test = reshape_data_for_t10(test)

    print("Shape of x_train: ", x_train.shape) 
    print("Shape of y_train: ", y_train.shape)
    print("Shape of x_test: ", x_test.shape) 
    print("Shape of y_test: ", y_test.shape)

    return x_train, y_train, x_test, y_test


def reshape_data_for_t10(data, sequence_length=10, group_size=100):
    """
    Restructure the input data for the VAE so that targets are 10 time steps ahead of inputs

    Parameters
    ----------
    data : ndarray
        Input data.
    sequence_length : int, optional
        Length of the sequence, by default 10.
    group_size : int, optional
        Size of the group, by default 100.

    Returns
    -------
    tuple
        Reshaped input and target data.
    """
    inputs = []
    targets = []

    for i in range(0, len(data), group_size):

        for j in range(i, i + group_size - sequence_length):
            inputs.append(data[j])
            targets.append(data[j + sequence_length])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def divide_dataset(data, num_splits):
    """
    Divide the dataset into a given number of splits. Used to split different wildfire simulations.

    Parameters
    ----------
    data : ndarray
        Input data.
    num_splits : int
        Number of splits.

    Returns
    -------
    list
        A list of datasets after division.
    """
    datasets = np.array_split(data, num_splits)
    return datasets


def split_and_shift_datasets(x_train, y_train, x_test, y_test, background):
    """
    Split and shift the datasets for creating final train, test, and background data for the LSTM model

    Parameters
    ----------
    x_train : ndarray
        Training dataset.
    y_train : ndarray
        Targets for the Training dataset.
    x_test : ndarray
        Test dataset.
    y_test : ndarray
        Targets for the Test dataset.
    background : ndarray
        Background data.

    Returns
    -------
    tuple
        Final Training and Test datasets, and Background data after shifting.
    """
    x_train_datasets = divide_dataset(x_train,125)
    y_train_datasets = divide_dataset(y_train,125)

    print("create final train data for LSTM")
    x_train_fnl, y_train_fnl = shift_train_data_for_10_steps(x_train_datasets, y_train_datasets)
    print("create final test data for LSTM")
    x_test_fnl, y_test_fnl = shift_test_data_for_10_steps(x_test, y_test)

    print("create final background data")
    background_fnl = shift_background_data(background)
    
    return x_train_fnl, y_train_fnl, x_test_fnl, y_test_fnl, background_fnl


def shift_train_data_for_10_steps(x, y):
    """
    Shift training data for the LSTM so y is 10 time steps ahead of x

    Parameters
    ----------
    x : ndarray
        Training dataset.
    y : ndarray
        Targets for the Training dataset.

    Returns
    -------
    tuple
        Training dataset and targets after shifting.
    """
    x_fnl = []
    for i in x:
        new_data = []
        for j in range(np.shape(i)[0]-20):
            new_data.append([i[j],i[j+10]])
        new_data = np.array(new_data)
        x_fnl.append(new_data)
   
    y_fnl = []
    for i in y:
        new_data =[]
        for j in range(np.shape(i)[0]-20):
            new_data.append(i[j+20])
        new_data = np.array(new_data)
        y_fnl.append(new_data)

    print('Shape of x_train_fnl:',np.shape(x_fnl))
    print('Shape of y_train_fnl:',np.shape(y_fnl))

    return x_fnl, y_fnl


def shift_test_data_for_10_steps(x_test,y_test):
    """
    Shift training data for the LSTM so y is 10 time steps ahead of x

    Parameters
    ----------
    x_test : ndarray
        Test dataset.
    y_test : ndarray
        Targets for the Test dataset.

    Returns
    -------
    tuple
        Test dataset and targets after shifting.
    """
    x_test_fnl = []
    for i in range(len(x_test)-20): 
        x_test_fnl.append([x_test[i], x_test[i+10]])  # Create pairs of (t=n, t=n+10)

    x_test_fnl = np.array(x_test_fnl)  

    y_test_fnl = []
    for i in range(len(y_test)-20):
        y_test_fnl.append(y_test[i+20])

    y_test_fnl = np.array(y_test_fnl)

    print('Shape of x_test_fnl:', x_test_fnl.shape)
    print('Shape of y_test_fnl:', y_test_fnl.shape)

    return x_test_fnl, y_test_fnl


def shift_background_data(background):
    """
    Shift background data for the LSTM so [t-1 and t-2] can be fed into the LSTM

    Parameters
    ----------
    background : ndarray
        Background data.

    Returns
    -------
    ndarray
        Background data after shifting.
    """
    back_fnl = []
    for i in range(len(background)-2):
        back_fnl.append([background[i],background[i+1]])
    back_fnl = np.array(back_fnl)
    print('Shape of back_fnl:', back_fnl.shape)
    return back_fnl