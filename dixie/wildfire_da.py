import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import inv

class DAwrapper:

    """
    A class to wrap the functions for Data Assimilation.
    
    """
    def __init__(self, predictions, satellite):
        """
        Initialise a new instance of the DAwrapper class.

        Parameters:
        - np.array predictions: An array of the prediction data from either the LSTM or the VAE models.
        - np.array satellite: An array of the wildfire satellite data used for Data Assimilation.
    
        """
        self.predictions =np.reshape(predictions,(len(predictions),256*256))
        self.satellite = np.reshape(satellite,(len(predictions),256*256))

    def compress(self,variance):
        """
        Compresses the prediction and satellite data to a specified target variance using PCA.

        Parameters:
        - float variance: The desired target variance for the reduced space

        """
        self.pca = PCA(variance)
        self.predictions_comp = self.pca.fit_transform(self.predictions)
        self.satellite_comp = self.pca.transform(self.satellite)
        self.minPCs = self.predictions_comp.shape[1]
    
    def reconstruct(self,data_comp):
        """
        Expands compressed data in the latent space to the real physical space.
        Can only be run after the compress() function has been run.

        Parameters:
        - np.array data_comp: An array of the compressed data to expand.
        
        Returns:
        - np.array data_recon: An array of the reconstructed data in the physical space.

        """
        data_recon = self.pca.inverse_transform(data_comp)
        return data_recon

    def _update_prediction(self, x, K, H, y):
        """
        Perform Data Assimilation on each timestep of prediction data using the satellite data.

        Parameters:
        - np.array x: An array of the prediction data at a specific timestep.
        - np.array K: An array of the Kalman Gain.
        - np.array H: An array of the Observational Operator.
        - np.array y: An array of the satellite data at a specific timestep.

        Returns:
        - np.array res: An array of the updated predictions after Data Assimilation.
        
        """
        res = x + np.dot(K,(y - np.dot(H, x)))
        return res  

    def _KalmanGain(self, B, H, R):
        """
        Calculate the Kalman Gain according to the covariance matrices and the Observational Operator.

        Parameters:
        - np.array B: An array of the satellite data covariance matrix.
        - np.array H: An array of the Observational Operator, typically the Identity matrix is used.
        - np.array R: An array of the prediction data covariance matrix.

        Returns:
        - np.array res: An array of the Kalman Gain.

        """
        tempInv = inv(R + np.dot(H,np.dot(B,H.transpose())))
        res = np.dot(B,np.dot(H.transpose(),tempInv))
        return res

    def assimilate(self):
        """
        Perform Data assimilation by iteratively updating the prediction data at each timestep

        """
        I = np.identity(self.minPCs)
        R = np.cov(self.satellite_comp.T)
        H = I 
        B = np.cov(self.predictions_comp.T) 
        K = self._KalmanGain(B, H, R) 
        updated_data_list = []
        for i in range(len(self.predictions_comp)):
            updated_data = self._update_prediction(self.predictions_comp[i], K, H, self.satellite_comp[i]) 
            updated_data_list.append(updated_data)
        updated_data_array = np.array(updated_data_list)

        self.predictions_assimilated = updated_data_array
