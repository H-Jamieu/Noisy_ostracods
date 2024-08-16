import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from utils import hoc, knn
import time
from tqdm import tqdm
from pathlib import Path

"""
Custom adaption of simi-feat from the doata.ai implementation.
"""

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Feature_dataset(Dataset):
    """
    DataLoader class. Sub-class of torch.utils.data Dataset class. It will load data from designated files.
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Initial function. Creates the instance.
        :param annotations_file: The file containing image directory and labels (for train and validation)
        :param img_dir: The directory containing target images.
        :param transform: Transformation applied to images. Should be a torchvision.transform type.
        :param target_transform:
        """
        # As feature file is large, the csv may take long time to load.
        print("Loading data from csv file... It may take a while.")
        self.img_features = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # for normal loading
        self.img_labels = self.img_features[[0,1]]
        self.label = self.img_features[1].values
        self.feature = self.img_features.iloc[:, 2:].values
        self.index = np.arange(len(self.img_features), dtype=int)

    def __len__(self):
        return len(self.img_labels.index)

    def __classes__(self):
        return max(self.img_labels[1].unique())+1

    def __getitem__(self, idx):
        """
        Controls what returned from data loader
        :param idx: Index of image.
        :return: image: The image array.
        label: label of training images.
        img_path: path to the image.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]
        return image, label, idx
    
    def get_features_from_csv(self, feature_file):
        """
        Get features from csv file.
        :param feature_file: The file containing features.
        :return: features: The features.
        """
        features = pd.read_csv(feature_file, header=None)
        return features
    

def simifeat(num_epoch, dataset, num_classes, hoc_cfg, method, show_details=True, k=10):
    # A data-centric method
    # Adapted from the paper: Z. Zhu, Z. Dong, Y. Liu. Detecting Corrupted Labels Without Training a Model to Predict. ICML'22.
    # Code: https://github.com/UCSC-REAL/SimiFeat
    # Paper: https://proceedings.mlr.press/v162/zhu22a/zhu22a.pdf
    # Original code: https://github.com/Docta-ai/docta/tree/master
    """
    Sample detect_cfg:
    detect_cfg = dict(
        num_epoch = 21,
        sample_size = 35000,
        k = 10,
        name = 'simifeat',
        method = 'rank'
    )
    """
    print(f'Detecting label errors with simifeat.')
    sel_noisy_rec = np.zeros((num_epoch, len(dataset)))
    sel_times_rec = np.zeros(len(dataset))
    suggest_label_rec = np.zeros((len(dataset), num_classes))
    T_given_noisy = None
    if method == 'rank': # estimate T with hoc (by default)
        # compute HOC
        T_est, p_est, T_est_before_sfmx = hoc.estimator_hoc(num_classes, hoc_cfg, dataset, show_details=True)
        noisy_posterior = np.array([sum(dataset.label == i) for i in range(num_classes)]) * 1.0
        noisy_posterior /= np.sum(noisy_posterior)
        T_given_noisy = T_est * p_est / noisy_posterior
        
        if show_details:
            print("T given noisy:")
            print(np.round(T_given_noisy, 2))

    print(f'Use SimiFeat-{method} to detect label errors.')
    time0 = time.time()
    for epoch in tqdm(range(num_epoch)):
        if show_details:
            print(f'Epoch {epoch}. Time elapsed: {time.time() - time0} s')
        # slighly different from the original paper. Its using subset on the dataset to introduce randomness. to KNN
        sel_noisy, sel_idx, suggest_label = knn.simi_feat_batch(num_classes, hoc_cfg, dataset, T_given_noisy, k, method)
        sel_noisy_rec[epoch][np.asarray(sel_noisy)] = 1
        sel_times_rec[np.asarray(sel_idx)] += 1
        suggest_label_rec[np.asarray(sel_noisy), suggest_label] += 1
        
    noisy_avg = (np.sum(sel_noisy_rec, 0) + 1) / (sel_times_rec + 2)
    # sel_clean_summary = np.round(1.0 - noisy_avg).astype(bool)
    sel_noisy_summary = np.round(noisy_avg).astype(bool)
    num_label_errors = np.sum(sel_noisy_summary)
    print(f'[SimiFeat] We find {num_label_errors} corrupted instances from {sel_noisy_summary.shape[0]} instances')
    idx = np.argsort(noisy_avg)[-num_label_errors:][::-1] # raw index
    suggest_matrix = (suggest_label_rec + 1) / (np.sum(suggest_label_rec, 1).reshape(-1,1) + num_classes) # #samples * #classes

    # update report
    detection = dict(
        label_error = [[i, noisy_avg[i]] for i in idx]
    )

    suggest_matrix[range(len(suggest_matrix)), np.array(dataset.label)] = -1
    curation = dict(
        label_curation = [[i, np.argmax(suggest_matrix[i]), suggest_matrix[i][np.argmax(suggest_matrix[i])] * noisy_avg[i]] for i in idx]
    )
    return detection, curation

def post_processing(dataset, label_curation):
    """
    Post processing for the detected label errors.
    :param dataset: The dataset.
    :param label_curation: The label curation.
    :return: dataset: The dataset after post processing.
    """
    problem_idx = label_curation[:, 0]
    all_images = dataset.img_labels
    noisy_images = all_images.iloc[problem_idx]
    problem_df = pd.DataFrame(label_curation, columns=['idx', 'label_correction', 'confidence'])
    problem_df.set_index('idx', inplace=True)
    noisy_images = pd.merge(noisy_images, problem_df, left_index=True, right_index=True)
    noisy_images = noisy_images[noisy_images['confidence'] > 0.2] # threshold taken from docta.ai implementation
    return noisy_images

def main(embedding_file, base_img_path, method):
    # for test only
    #embedding_file = 'D:/Noisy_ostracods/datasets/embeddings/b_32_embed.csv'
    feature_dataset = Feature_dataset(embedding_file, base_img_path)
    num_epoch = 35 # Expected every sample to be corrupted at least 5 times
    num_classes = feature_dataset.__classes__()
    hoc_cfg = dict(
        max_step = 1501, #default 1501 
        T0 = None, 
        p0 = None, 
        lr = 0.1, 
        num_rounds = 50, # default 50
        sample_size = int(0.7 * len(feature_dataset)),
        already_2nn = False,
        device = 'cuda:1'
    )
    hoc_cfg = Config(**hoc_cfg)
    detection, curation = simifeat(num_epoch, feature_dataset, num_classes, hoc_cfg, method)
    label_error = np.array(detection['label_error'])
    label_curation = np.array(curation['label_curation'])
    # intermediate code
    # sel = label_curation[:, 2] > 0.2
    # cured_labels = label_curation[sel, 1].astype(int)
    # save_path = 'cured_labels_ostracods_genus.npy'
    # np.save(save_path, cured_labels)
    # print(f'Saved cured labels to {save_path}')
    # np.save('label_error_ostracods_genus.npy', label_error)
    # print(f'Saved label errors to label_error_ostracods_genus.npy')
    # np.save('label_curation_ostracods_genus.npy', label_curation)
    # print(f'Saved label curation to label_curation_ostracods_genus.npy')
    out_df = post_processing(feature_dataset, label_curation)
    embedding_stem_name = Path(embedding_file).stem
    out_name = 'analyses/'+embedding_stem_name+f'simifeat_{method}_label_errors.csv'
    out_df.to_csv(out_name, index=False, header=None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, help='Path to the embedding file.')
    parser.add_argument('--base_img_path', type=str, help='Path to the base image directory.', default='D:/ostracods_data/class_images')
    parser.add_argument('--method', type=str, help='method of simifeat', default='rank')
    args = parser.parse_args()
    main(args.embedding_file, args.base_img_path, args.method)