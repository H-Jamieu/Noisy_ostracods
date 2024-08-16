import pandas as pd
import numpy as np
import os
import regex as re
from cleanlab import Datalab

"""
This script is for using the cleanlab to clan the labels using the activations from the cross_validation models.
As we already computed the activations from crossvalidation, we just use some sub-parts of the clean lab.
"""

def find_activations_file(root_folder, pattern):
    """
    Fine the activations with the desired pattern in the root folder and read them into a dataframe.
    :param root_folder: str, the root folder to search for the activations.
    :param pattern: regex pattern, the pattern to search for in the root folder.
    """
    activations = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if bool(re.search(pattern, file)):
                print(f"Found activations file: {file}")
                activation = pd.read_csv(os.path.join(root, file), header=None)
                activations.append(activation)
    # concat the activations
    binded_activations = pd.concat(activations, axis=0)
    # reassign the index
    binded_activations = binded_activations.reset_index(drop=True)
    print(f"Found {len(binded_activations)} activations.")
    return binded_activations

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def hard_softmax(x):
    for i in range(len(x)):
        x[i] = softmax(x[i])
    return x

def find_label_issues(activations):
    """
    Find the labels that have issues using cleanlab.
    """
    data_with_labels = activations[[0,1]]
    data_with_labels.columns = ['file_name', 'labels']
    lab = Datalab(data_with_labels, label_name='labels')
    features = activations.iloc[:, 3:].to_numpy()
    preds = hard_softmax(features)
    lab.find_issues(pred_probs=preds, issue_types={"label": {}})
    print(lab.get_issue_summary("label"))
    confident_joint = lab.get_info("label")['confident_joint']
    return lab.get_issues('label'), confident_joint

def out_put_keys(activations, issues, confident_joint, pattern, output_folder):
    """
    Output the keys for the labels that have issues.
    """
    data_with_labels = activations[[0,1]]
    data_with_labels.columns = ['file_name', 'labels']
    if 'genus' in pattern:
        pattern = 'ostracods_genus'
    else:
        pattern = 'ostracods_species'
    conbined = pd.concat([data_with_labels, issues], axis=1)
    save_name_combined = os.path.join(output_folder, f'{pattern}_all_label_issues.csv')
    confident_joint_savename = os.path.join(output_folder, f'{pattern}_confident_joint.npy')
    conbined.to_csv(save_name_combined, index=False)
    np.save(confident_joint_savename, confident_joint)
    print(f"Output the keys to {save_name_combined} and {confident_joint_savename}")

def main():
    patterns = [r"ostracods_genus_cv\d+_resnet", r"ostracods_species_cv\d+_resnet"]
    root_folder = 'your activation path' # your folder saving the activations
    save_folder = './analyses' # saving the outputs
    for pattern in patterns:
        activations = find_activations_file(root_folder, pattern)
        issues, confident_joint = find_label_issues(activations)
        out_put_keys(activations, issues, confident_joint, pattern, save_folder)

if __name__ == '__main__':
    main()