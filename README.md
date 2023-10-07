# Wildfire project

## Deadlines

- Code due 12:00 BST Friday 26th May
- Presentation 17:00 BST Friday 26th May

## Setup

```
conda env create -f environment.yml
```

### MacOS M1 setup

```
conda create --name dixie tensorflow
```

### activate the env
```
conda activate dixie
```


## Running our models

Our final models are run in the notebook ```dixie/Dixie.ipynb```

This notebook must be run in Colab using A100 GPUs. 
The Ferguson fire data must be uploaded into your google drive, and keep them all in the same folder. The path to this folder can be specified in the notebook when loading in the data.
Currently it is set to ```/content/drive/MyDrive/```

This notebook uses python classes to do preprocessing and to load and train the models. Please make sure that all the files in the ```dixie/``` folder are in your drive as well, in the same folder as the notebook.

### optional: run with pretrained LSTM

In the ```Dixie.ipynb``` notebook, you can specify a .keras model. We have uploaded our optimal LSTM model here: https://imperiallondon-my.sharepoint.com/:u:/g/personal/cs1622_ic_ac_uk/EUK6xbd-CkdHr8Ku-OGQ2F4BphPoUmLthzw-tcFaaS7Zkw?e=SDzenR

The LSTM model takes a long time to train, but it shouldnt be so bad with the A100 GPUs.

The VAE takes less time, so we have not uploaded the model weights for the VAE.


## Other files

Our repository contains another folder called ```dev_notebooks```. These notebooks contain the latest experiments run for all 3 objectives. We have consolidated the optimal models from these notebooks into the python classes for each one.


## Run tests

Pytests can be run locally. Clone the repository and within the dixie conda environment, run:
```
pytest
```
