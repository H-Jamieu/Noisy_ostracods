import time
import numpy as np

import torch
import torch.nn.functional as F


"""
Adapted from the docta.ai knn.py. Removed the dependency on cfg.
Original: https://github.com/Docta-ai/docta/blob/master/docta/core/knn.py
"""

# all from hoc.py, to avoid nested import as when import knn, we must import hoc
def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix

def get_consensus_patterns(dataset, sample, k=1+2):
    """ KNN estimation
    """
    feature = dataset.feature if isinstance(
        dataset.feature, torch.Tensor) else torch.tensor(dataset.feature)
    label = dataset.label if isinstance(
        dataset.label, torch.Tensor) else torch.tensor(dataset.label)
    feature = feature[sample]
    label = label[sample]
    dist = cosDistance(feature.float())
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    knn_labels = label[indices]
    return knn_labels, values

def count_knn_distribution(num_classes, dataset, sample, k=10, norm='l2', show_details=True):
    """ Count the distribution of KNN
    Args:
        cfg: configuration
        dataset: the data for estimation
        sample: the index of samples
        k : the number of classes
    """

    time1 = time.time()
    num_classes = num_classes
    knn_labels, values = get_consensus_patterns(dataset, sample, k=k)
    # make the self-value less dominant (intuitive)
    # since the closed neighbors is the data itself, a normlization is needed
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]

    knn_labels_cnt = torch.zeros(len(sample), num_classes)

    for i in range(num_classes):
        # knn_labels_cnt[:,i] += torch.sum(1.0 * (knn_labels == i), 1) # not adjusted
        # adjusted based on the above intuition
        knn_labels_cnt[:,
                       i] += torch.sum((1.0 - values) * (knn_labels == i), 1)

    time2 = time.time()
    if show_details:
        print(f'Running time for k = {k} is {time2 - time1} s')

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / \
            torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob


def get_score(knn_labels_cnt, label):
    """ Get the corruption score. Lower score indicates the sample is more likely to be corrupted.
    Args:
        knn_labels_cnt: KNN labels
        label: corrupted labels
    """
    score = F.nll_loss(torch.log(knn_labels_cnt + 1e-8),
                      label, reduction='none')
    return score


def simi_feat_batch(num_classes, hoc_cfg, dataset, T_given_noisy, k=10, method='mv'):
    """ Construct the set of data that are likely to be corrupted.
    Example hoc_cfg:
    hoc_cfg = dict(
        max_step = 1501, 
        T0 = None, 
        p0 = None, 
        lr = 0.1, 
        num_rounds = 50, 
        sample_size = 35000,
        already_2nn = False,
        device = 'cpu'
    )
    T_given_noisy: the first return from hoc 
    """

    # Build Feature Clusters --------------------------------------

    sample_size = int(len(dataset) * 0.9)
    if hoc_cfg is not None and hoc_cfg.sample_size:
        sample_size = np.min((hoc_cfg.sample_size, int(len(dataset)*0.9)))

    # different from the original implementation, the randomness is not based on the feature argumentation but randomness in the sample selection
    idx = np.random.choice(range(len(dataset)), sample_size, replace=False)

    knn_labels_cnt = count_knn_distribution(
        num_classes=num_classes, dataset=dataset, sample=idx, k=k, norm='l2')

    score = get_score(knn_labels_cnt, torch.tensor(dataset.label[idx]))
    score_np = score.cpu().numpy()
    sel_idx = dataset.index[idx]  # raw index

    label_pred = np.argmax(knn_labels_cnt.cpu().numpy(), axis=1).reshape(-1)
    if method == 'mv':
        # test majority voting
        # print(f'Use MV')
        sel_true_false = label_pred != dataset.label[idx]
        sel_noisy = (sel_idx[sel_true_false]).tolist()
        suggest_label = label_pred[sel_true_false].tolist()
    elif method == 'rank':
        # print(f'Use ranking')

        sel_noisy = []
        suggest_label = []
        for sel_class in range(num_classes):
            # setting fix rule to elimate negative class
            if sel_class == 42 or sel_class == 84:
                if num_classes <= 84:
                    continue
                else:
                    if sel_class == 84:
                        continue
            thre_noise_rate_per_class = 1 - \
                min(1.0 * T_given_noisy[sel_class][sel_class], 1.0)
            # clip the outliers
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = dataset.label[idx] == sel_class
            if np.sum(sel_labels) == 0:
                # due to imbalance, some classes may not be selected
                continue
            thre = np.percentile(
                score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
            suggest_label += label_pred[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    # raw index, raw index, suggested true label
    return sel_noisy, sel_idx, suggest_label
