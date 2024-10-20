# Noisy_ostracods
 The officical implementation of collected methods in the Noisy Ostracods dataset. We are now refactoring the code to make it easier to use. Here is queue of works:

 1. Implement Dynamic Loss for Robust Learning (Jiang et al., 2023) which desigened specifically for unbalanced Label Noise Learning. (Done)
 2. Integrating Dividemix, SDN and PLM into out trainer->train_engine structure. However, all mthdods included subseting the dataset during train time, a seprate engine will be developed.
 3. Implement a unifed calling interface for both robust training and label correction methods.
 4. Write documentations.
 
# Downloads
To be released

Data Documentation: 
# Data and label
The data is provided in the zip file from the download links. When you can unzip the file and put it in any directory you like. We represent the directory as `Root Folder`. The file structure after unzipping should be looks like:
```
-Root Folder
|---Noisy_ostracods
    |--sinocythere sinensis
        |--images
    |---other species
        |--images
```
The label file is csv file with rows looks like `aurila cfdisparata rawSample_B3b_19_ind1.tif, 5`. The first part before comma is file relative path after knowing `Root Folder`. The second part in the index number in the guidance files provided in the `datasets` folder of this repository. We provided the label file in format of `ostracods_{target}_{project}_{phase}.csv`. The `{target}` can be `genus`, `species` and even `family` given the hierarchical structure of the taxonomy. The `{phase}` can be `train`, `val` and `test` reference to different phases of the machine learning. The train:validation:test ratio for this study is controlled to be 8:1:1.

For `project`, the provided label files are:

1. `final`: The original noisy labels of the noisy ostracods dataset.
2. `CL`: The filtered images by Confident Learning (Northcutt et al., 2021).
3. `clean`: The cleaned labels by experts. Still updating.
4. `trans`: The noisy labels of just 51+1 classes inside the test set. It is for utlizing the 52*52 transistion matrix for testing the efficiency of transistion matrix based methods.

The label files and guidance files for 2022 version is also avaliable within `datasets/2022.zip`. The usage is the same as current version.
# Project file organizations
The training of `co-teaching`, `co-teaching+`, `Cross Entropy`, `Loss-clip` and `transition-matrix` are packed in similar calling procedure. To understand how to train the models, we need to under standing the project structure.

```
Root Folder
|--The_Noisy_Ostracods_projectfile
   |--analyses
   |--ckpt
   |--datasets
   |--log_dir
   |--utils
   codes
   config.yaml
   ...
```
The folder structures looks like the above file tree. The `analyses` folder contains the analyses files such as the transistion matrixes. The `ckpt` folder is the directory for saving trained models. `dataset` folder contains the label files. `log_dir` is directory for saving the training logs. `utils` is for putting utility codes.
In the main directory, the `trainer.py` is the entry for training the models mentioned in the begining. We just modify the entries in the `config.yaml` to achieve the training target. For example, by default, the given `config.yaml` is training `transition-matrix` using `resnet_50` backbone.

# Cross Entropy
Modify the `config.yaml` file to change the `task` to `vanilla`. Chnage the backbone in the `model` entry to your preferred backbone. By default, only `resnet50` and `vit-b-16` are provided, but you can add other backbones according to your preference. You can also adjust other hyper-parameters such as `epochs`, `base_learning_rate`, and `scheduler` as needed.

The dataset settings can be controlled by modifying `target`, `class_img_path`, and `base_path`. The code retrieves the image paths by using the entries in the label file specified by `target`, and then constructs the absolute path by combining `base_path`, `class_img_path`, and the image entry in the label file. For example, if the Noisy Ostracods images are located in the `Root Folder`, set `base_path` to `Root Folder` and `class_img_path` to `the_noisy_ostracods`. By setting `target` to `ostracods_genus_final`, you can train a Cross Entropy `resnet50` model using the noisy training data.

After make all chnage, run `python trainer.py`. The training will start, and results will be saved in `ckpt` folder. A copy of the config file will be saved to `log_dir` for recording.

# Coteaching
Modify the `config.yaml` file to change the `task` to `co-teaching`. Setting the two models by chnage the entries `model` and `co-model`. Change the forget-rate by modifying `forget_rate` entry. Change other hyper-parameters accordingly.

After make all chnage, run `python trainer.py`. The training will start, and results will be saved in `ckpt` folder. A copy of the config file will be saved to `log_dir` for recording.

# Coteaching Plus

Modify the `config.yaml` file to change the `task` to `co-teaching+`. Setting the two models by chnage the entries `model` and `co-model`. Change the forget-rate by modifying `forget_rate` entry. Change other hyper-parameters accordingly.

After make all chnage, run `python trainer.py`. The training will start, and results will be saved in `ckpt` folder. A copy of the config file will be saved to `log_dir` for recording.

# Loss-clip
Modify the `config.yaml` file to change the `task` to `mentor`. Setting the model by chnage the entry `model`. Change the forget-rate by modifying `forget_rate` entry. Change other hyper-parameters accordingly.

After make all chnage, run `python trainer.py`. The training will start, and results will be saved in `ckpt` folder. A copy of the config file will be saved to `log_dir` for recording.

# Divide-mix
```python train_N_ostracods.py --data_path={your ostracod image path} --batch_size={according to you GPU} --lr={according to your batch size}```

By default, the `ostracods_genus_final` is used for training. If you want to change the data, just chnage the relvant lines in `utils\dataloader_N_ostracods.py`.

Original author: https://github.com/LiJunnan1992/DivideMix

# Simi-feat (Zhu et al., 2022)
1. Download the data. If you want to use our pre-computed embeddings, go to 3. Pre-computed embeddings in: https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiamianh_connect_hku_hk/EpJIOKpi-CBNvJ7vBSZtsVUBpi8kxMDwMT5VwDV8wSCC4g?e=8oe4CL. If you want to compute your own embedding, go to 2.
2. Compute embedding using `utils/compute_embeddings.py`. The script supports CLIP (Radford et al., 2021) and Masked-AutoEncoder (He et al., 2021) embedding. Specify your prefreed method by calling the respective embedding function inside the script. 
 
3. `python simifeat_entry.py --embedding_file={your embedding file path} --base_img_path={your ostracod image path} --mtehod={rank or mv}`

# Confident Learning
Adapted from https://github.com/cleanlab/cleanlab.
```
python cl_entry.py
```
Please change the activation saving path by change the code in `cl_entry.py` accordingly. The percomputed CL noisy data are provided in the `analyses` folder. They are `ostracods_genus_all_label_issues.csv` and `ostracods_species_all_label_issues.csv`.
Download the activations: https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiamianh_connect_hku_hk/EtV6uznsF2xGjwYIzfBqcA4B9-nBO_WRf9QuhyL1Iki5zQ?e=oK8z2Y
# Naive-Ensemble-Cross-Validation

The Naive-Ensemble-Cross-Validation (NECV) is straitforward.
1. Split dataset into 10 random sets. Named cv1, cv2 ...., cv10 in the `project` postfix of `ostracods_{target}_{project}_{phase}.csv`.
2. Train the model on the data excluding the splits. Valid on the corresponding split.
3. Recording the inference result. Caculate the agreement ratio as the number of predictions agree with the original label given.
4. Filter data based on the agreement ratio. We set 0.5.

Our splits could be downloaded at https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiamianh_connect_hku_hk/EtZZDPGtvN5JucNCqapTxFMB23SJ6P_7W4OQTPs9cOjcQg?e=FjSZyZ.

# Evaluation
Change `config_evaluation.yaml` changing the `model_path` entry to your trained model. The specifying the `model` to the model type of your model e.g. resnet50. Then run evaluation.py, we will get Accurancy, Precision, Recall and F1-Score for the models on the target dataset.
# Results
## Robust machine learning
P@ is precision, R@ is Recall, F1@ is F1 Score.

| Method           | Acc     | P@    | R@    | F1@   | Acc     | P@    | R@    | F1@   |
|------------------|---------|-------|-------|-------|---------|-------|-------|-------|
|                  | **ResNet-50** |       |       |       | **ViT-B-16** |       |       |       |
| CE               | 95.98   | **88.50** | **77.80** | **79.51** | **95.03** | **83.31** | **75.30** | **76.64** |
| Mixup+Cutmix     | 95.11   | 74.96 | 69.83 | 69.42 | 90.33   | 57.99 | 51.95 | 52.98 |
| CL               | 95.16   | 80.97 | 69.01 | 71.51 | 90.62   | 57.48 | 53.27 | 53.75 |
| Loss-clip        | 95.79   | 81.69 | 72.78 | 74.09 | 93.71   | 72.14 | 64.27 | 65.26 |
| Co-teaching      | 95.79   | 79.01 | 71.79 | 73.23 | 94.71   | 77.77 | 71.69 | 72.37 |
| Co-teaching+     | **96.19** | 84.09 | 77.57 | 78.21 | 91.82   | 66.04 | 63.45 | 63.65 |
| Divide-mix       | 95.50   | 53.42 | 57.33 | 54.86 | 84.78   | 29.99 | 28.21 | 24.92 |

## Label cleanning
P. is short for `Paradoxostomaid`. Pesudo class is `Cyperdeis`.
| Method               | Hit-rate | Feature error hit-rate | Label error hit-rate | P@    | Found pseudo classes | $P.$ hit-rate | Hit-rate w/o. $P.$ |
|----------------------|----------|------------------------|-----------------------|-------|-----------------------|---------------|--------------------|
| CL                   | 59.37    | 62.95                  | 54.06                 | **64.13** | F                     | 7.94          | 75.41              |
| SimiFeat-CLIP        | 26.29    | 30.11                  | 20.63                 | 17.13 | F                     | 6.88          | 32.34              |
| SimiFeat-CLIP-mv     | 26.29    | 29.05                  | 22.19                 | 17.29 | F                     | 7.41          | 32.18              |
| SimiFeat-MAE-fnt     | 28.18    | 30.32                  | 25.00                 | 47.16 | T                     | 3.17          | 35.97              |
| SimiFeat-MAE-fnt-mv  | 31.07    | 32.21                  | 29.38                 | 44.67 | T                     | 2.65          | 39.93              |
| SimiFeat-MAE-pre     | 27.55    | 32.21                  | 20.63                 | 13.83 | T                     | 12.70         | 32.18              |
| SimiFeat-MAE-pre-mv  | 26.79    | 32.00                  | 19.06                 | 13.87 | T                     | **14.29**     | 30.69              |
| NECV                 | **71.19** | **80.42**              | **57.50**             | 55.87 | T                     | 5.82          | **91.58**          |
# Chnaging dataset
By design, any image dataset save in image fomat with a `.csv` label file would work for all experiments. You can use this as a tool to test if those data cleanning method would work in your own dataset.