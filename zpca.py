import argparse
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from basicts.data.transform import standard_transform

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))

if __name__ == '__main__':
    DATASET_NAME = "METR-LA"
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_NAME)
    TARGET_CHANNEL = [0]                   # target channel(s)
    data_file_path = DATA_FILE_PATH
    target_channel = TARGET_CHANNEL
    df = pd.read_hdf(data_file_path)
    # data = np.expand_dims(df.values, axis=-1)

    # data = data[..., target_channel]
    # print("raw time series shape: {0}".format(data.shape))

    data = df.values
    print("raw time series shape: {0}".format(data.shape))
    # print(data)
    # print(np.max(data,axis=0))
# PCA
    # d = 32 # hyper-parameters: dimension
    # pca = PCA(n_components=d)
    pca = PCA()
    pca.fit(data)
    data_pca = pca.transform(data)
    print(np.max(data_pca, axis=0))
    print(data_pca.shape)
#     print("pca time series shape: {0}".format(data_pca.shape))

# # noised
    mu = 0  # hyper-parameters: mean=0
    sigma = 0.2  # hyper-parameters: standard deviation
    noise_matrix = np.random.normal(mu, sigma, data_pca.shape)
    data_pca += noise_matrix


# inv PCA
    data_inv_pca = pca.inverse_transform(data_pca)
    print("inv pca time series shape: {0}".format(data_inv_pca.shape))
    print(np.max(data_inv_pca, axis=0))
