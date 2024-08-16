import os
import pandas as pd
import numpy as np
from scipy import stats

"""
    knn-classifier using the nn_computed from clip-embeddings
"""
def knn_classifier(knn_array, k=1):
    """
        knn_array: numpy array of shape (n_samples, n_neighbors), all integers
        k: number of neighbors to consider
        returns: numpy array of shape (n_samples, 1)
    """
    return stats.mode(knn_array[:, :k], axis=1)[0]

def preprocess_knn_file(knn_file):
    """
        preprocesses the knn_file
        The knn_file is dataframe of image_path, nn_0, nn_1, nn_2, ..., nn_n
        where nn_i is the i-th nearest neighbor of the image, as path format
        We need to do the following:
            1. Extrat the class from the dataframe, using split on '/'
    """
    len_col = len(knn_file.columns)
    clesses = knn_file[0].apply(lambda x: x.split('/')[0].split(' ')[0]).unique()
    # apply split to all columns except the first one
    for i in range(1, len_col):
        knn_file[i] = knn_file[i].apply(lambda x: x.strip().split('/')[0].split(' ')[0])
        # convert clasee to the corresponding index of clesses
        knn_file[i] = knn_file[i].apply(lambda x: clesses.tolist().index(x))
    return knn_file, clesses

if __name__ == '__main__':
    knn_file = pd.read_csv('vit_l_14_nn_31.csv', header=None)
    files = knn_file[0].values
    knn_file, classes = preprocess_knn_file(knn_file)
    # convert the row 1 after of knn_file to numpy array
    knn_array = knn_file.iloc[:, 1:].values
    knn_array = knn_array.astype(int)
    k_values = [5, 7, 11, 13, 17, 23, 31]
    all_pred = []
    for k in k_values:
        y_pred = knn_classifier(knn_array, k)
        y_pred = y_pred.flatten()
        txt_pred = [classes[i] for i in y_pred]
        all_pred.append(txt_pred)
    # save the predictions
    with open('vit_l_14_nn_31_pred.csv', 'w') as f:
        for i in range(len(files)):
            line = files[i]
            for pred in all_pred:
                line += ',' + pred[i]
            f.write(line + '\n')
    print('Done!')