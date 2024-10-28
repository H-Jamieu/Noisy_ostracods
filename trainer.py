import os
import pandas as pd
import torch
import time
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
from train_engine import train_model_coteaching, train_model_mentor, train_model_coteaching_plus, train_model, \
train_model_transition, train_model_SAM, train_model_AUM, train_model_LW, train_model_margin
import utils.customizedYaml as customizedYaml

'''
Sample Mean and Std in (r,g,b) from validation set
    mean: 0.119743854 0.12807578 0.23815322
    std: 0.113375455 0.112062685 0.22721237
Could increase accuracy
'''

sharing_strategy = "file_system"
MODEL_BACKUP_PTH = "ckpt"


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


class CustomImageDataset(Dataset):
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
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels.index)

    def __classes__(self):
        return len(self.img_labels[1].unique())

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


# Load train, test and validation data by phase. Phase = train, val and test. Target = genus and species
def load_data(phase, target, d_transfroms, batch_size=16, n_workers=0, cache=False, src_path='', shuffle=True):
    if len(phase) != 0:
        phase = '_' + phase
    data_path = './datasets/' + target + phase + '.csv'
    data_out = CustomImageDataset(data_path, src_path, d_transfroms)
    data_size = data_out.__len__()
    if n_workers == 0:
        data_loader = DataLoader(data_out, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle)
    else:
        data_loader = DataLoader(data_out, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle,
                                 pin_memory=True, timeout=120, worker_init_fn=set_worker_sharing_strategy)

    return data_loader, data_size


def define_transforms(img_sz, cut_mix = False):
    image_w = imgae_h = img_sz

    data_transforms = transforms.Compose([transforms.Resize([image_w, imgae_h]),
                                          transforms.RandomChoice([transforms.RandomRotation([90, 90]),
                                                                   transforms.RandomRotation([180, 180]),
                                                                   transforms.RandomRotation([270, 270])],
                                                                  p=[0.25, 0.25, 0.25]),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomChoice([transforms.ColorJitter(brightness=0.5),
                                                                   transforms.ColorJitter(brightness=0.5, hue=0.1)],
                                                                  p=[0.3, 0.3]),
                                          transforms.RandomGrayscale(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize([image_w, imgae_h]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    return data_transforms, valid_test_transforms


'''
Include the contents of entrance.py
'''

def preprocess_transition_matrix(trans_mat_path, device):
    
    trans_mat = np.load(trans_mat_path)
    # fill na with 52, the number of clasees to impose large penalty
    trans_mat = np.nan_to_num(trans_mat, nan=0)
    # iterate through the matrix, if diginal is 0, set it to 1
    for i in range(len(trans_mat)):
        if trans_mat[i][i] == 0:
            trans_mat[i][i] = 1
    # transfer to tensor, force to be float32
    trans_mat = torch.tensor(trans_mat, dtype=torch.float32).to(device)
    return trans_mat

def init_weights(layer):
    torch.nn.init.kaiming_uniform_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.normal_(layer.bias, std=1e-6)


def modify_model(model_txt, model_for_modify, class_num):
    resnet_family = ['resnet50']
    vit_family = ['vit_b_16']
    model_txt = model_txt.lower()
    if model_txt in vit_family:
        num_ftrs = model_for_modify.heads.head.in_features
        model_for_modify.heads.head = torch.nn.Linear(num_ftrs, class_num)
        init_weights(model_for_modify.heads.head)
    else:
        # assume to be in resnet family
        num_ftrs = model_for_modify.fc.in_features
        model_for_modify.fc = torch.nn.Linear(num_ftrs, class_num)
        init_weights(model_for_modify.fc)


def determine_model(arg_model, arg_pretrain):
    """
    Since the models have different implementations in the behavior of the last layer. We need to write the tedious
    implementation as below.
    Subject to refactor.
    """
    model_text = arg_model.lower()
    pre_train_flag = True if arg_pretrain == 'DEFAULT' else False
    if arg_model.lower() == 'vit_b_16':
        model_for_experiment = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) if pre_train_flag is True else \
            models.vit_b_16(weights=None)
        # num_ftrs = model_for_experiment.heads.head.in_features
        # model_for_experiment.heads.head = torch.nn.Linear(num_ftrs, arg_classes)
    else:
        model_text = 'resnet50'
        if arg_pretrain == 'DEFAULT':
            model_for_experiment = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            print('Using non-pretrained weights')
            model_for_experiment = models.resnet50(weights=None)
        # num_ftrs = model_for_experiment.fc.in_features
        # model_for_experiment.fc = torch.nn.Linear(num_ftrs, arg_classes)
    print(f'Using {model_text}. With pretrained set to be {str(pre_train_flag)}')
    return model_for_experiment


def determine_optimizer(model, arg_optimizer, arg_lr, arg_momentum, arg_lambda):
    if arg_optimizer.lower() == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=arg_lr, lambd=arg_lambda)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=arg_lr, momentum=arg_momentum)
    return optimizer


def determine_criterion(arg_criterion):
    return torch.nn.CrossEntropyLoss()


def determine_scheduler(optimizer, arg_scheduler, arg_step_size, arg_gamma):
    if arg_scheduler.lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg_step_size, gamma=arg_gamma)
    elif arg_scheduler.lower() == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=arg_gamma)
    elif arg_scheduler.lower() == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg_step_size, gamma=arg_gamma)
    return scheduler


def determine_device(device_no, FP16_flag='True'):
    model_scaler = None
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= int(device_no) + 1:
            device = torch.device('cuda:' + device_no)
            if FP16_flag == 'True':
                model_scaler = torch.cuda.amp.GradScaler()
        else:
            device = torch.device('cpu')
    elif torch.backends.mps.is_available():
        # Mac OS support
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device, model_scaler


if __name__ == '__main__':
    stage = 'Preparation'
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    yaml_file = customizedYaml.yaml_handler('config.yaml')
    args = yaml_file.data
    device, model_scaler = determine_device(args['device'], args['FP16'])
    print('Using ', device, '.')
    # Building hyperparameters
    model = determine_model(args['model'], args['weight'])
    modify_model(args['model'], model, args['classes'])
    model = model.to(device)
    lr = args['base_learning_rate'] * args['batch_size']
    optimizer = determine_optimizer(model, args['optimizer'], lr, args['momentum'], args['lambd'])
    scheduler = determine_scheduler(optimizer, args['scheduler'], args['step_size'], args['gamma']) 
    base_img_path = os.path.join(args['base_path'], args['class_img_path'])
    data_transforms, valid_test_transforms = define_transforms(args['imgsz'])
    warm_up = args['epochs'] // 10
    rate_schedule = np.ones(args['epochs'])*args['forget_rate']
    rate_schedule[:warm_up] = np.linspace(0, args['forget_rate'], warm_up)
    rho = args['rho']
    if args['val'] == 'F':
        effective_phase = ['train']
    else:
        effective_phase = ['train', 'val']
    try:
        # Warped the calling into single calling file to reduce multiprocessing error
        # Stage text will be passed to exception to indicate which stage is failed.
        stage = f"Training {args['model']} with {str(args['epochs'])} epochs for {args['target']}"
        dataloaders, dataset_sizes = [{}, {}]
        for phase in effective_phase:
            dataloaders[phase], dataset_sizes[phase] = load_data(phase, args['target'], data_transforms,
                                                                  args['batch_size'], args['workers'],
                                                                  False, base_img_path)
        # Warped the dataloader iteration to be closed by if __name__ == '__main__' clause accoring to
        # https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
        print(f'Current task is {args["task"]}')
        if args['task'] == 'co_teaching' or args['task'] == 'co_teaching+':
            co_model = determine_model(args['co_model'], args['weight'])
            modify_model(args['co_model'], co_model, args['classes'])
            co_model = co_model.to(device)
            co_optimizer = determine_optimizer(co_model, args['optimizer'], lr, args['momentum'], args['lambd'])
            co_scheduler = determine_scheduler(co_optimizer, args['scheduler'], args['step_size'], args['gamma'])
            if args['task'] == 'co_teaching':
                model, co_model, save_path = train_model_coteaching(model=model, co_model=co_model, co_optimizer=co_optimizer, optimizer=optimizer,
                scheduler=scheduler, co_scheduler=co_scheduler, num_epochs=args['epochs'], dataloaders=dataloaders,
                dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                effective_phase=effective_phase, rate_schedule=rate_schedule)
            else:
                model, co_model, save_path = train_model_coteaching_plus(model=model, co_model=co_model, co_optimizer=co_optimizer, optimizer=optimizer,
                scheduler=scheduler, co_scheduler=co_scheduler, num_epochs=args['epochs'], dataloaders=dataloaders,
                dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                effective_phase=effective_phase, rate_schedule=rate_schedule)

        elif args['task'] == 'mentor':
            model, save_path = train_model_mentor(model=model, optimizer=optimizer, scheduler=scheduler,
                                                  num_epochs=args['epochs'], dataloaders=dataloaders,
                                                  dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                  effective_phase=effective_phase, rate_schedule=rate_schedule)
        elif args['task'] == 'vanilla':
            model, save_path = train_model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=determine_criterion(args['criterion']),
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase)
        elif args['task'] == 'transition_matrix':
            cm_path = './analyses/cm_combined.npy'
            trans_mat = preprocess_transition_matrix(cm_path, device)
            model, save_path = train_model_transition(model=model, optimizer=optimizer, scheduler=scheduler, transition_matrix=trans_mat,
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase)
        elif args['task'] == 'SAM':
            model, save_path = train_model_SAM(model=model, optimizer=optimizer, scheduler=scheduler, criterion=determine_criterion(args['criterion']),
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase)
        elif args['task'] == 'AUM':
            
            model, save_path, new_records = train_model_AUM(model=model, optimizer=optimizer, scheduler=scheduler, criterion=determine_criterion(args['criterion']),
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase)
            save_dir = f'./log_dir/{save_path}_AUM.pth'
            new_records.save_aum_ranking(save_dir)
        elif args['task'] == 'LW':
            add_loader, _ = load_data(phase, args['target'], valid_test_transforms,
                                                                  args['batch_size'], args['workers'],
                                                                  False, base_img_path, shuffle=False)
            model, save_path = train_model_LW(model=model, optimizer=optimizer, scheduler=scheduler, criterion=determine_criterion(args['criterion']),
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase, add_loader=add_loader)
        elif args['task'] == 'margin':
            data_path = './datasets/ostracods_genus_clean_val.csv'
            data_meta = CustomImageDataset(data_path, base_img_path, valid_test_transforms)
            meta_loader = DataLoader(data_meta, batch_size=args['batch_size'], num_workers=8, 
                                     shuffle=False, pin_memory=True, timeout=120)
            dataloaders['meta'] = meta_loader
            model, save_path = train_model_margin(model=model, optimizer=optimizer, scheduler=scheduler, criterion=determine_criterion(args['criterion']),
                                                   num_epochs=args['epochs'], dataloaders=dataloaders,
                                                   dataset_sizes=dataset_sizes, device=device, scaler=model_scaler,
                                                   effective_phase=effective_phase, meta=True)
        else:
            raise Exception(f'Task not supported.')
        # Build model save path
        save_path = save_path + '_' + args['target'] + '_' + args['model'] + '.pth'
        model_path = os.path.join(args['save_folder'], save_path)
        torch.save(model.state_dict(), model_path)
        if args['task'] == 'co_teaching' or args['task'] == 'co_teaching+':
            co_save_path = save_path.replace(args['model'], args['co_model'])
            co_model_path = os.path.join(args['save_folder'], co_save_path)
            torch.save(co_model.state_dict(), co_model_path)
        save_time_stamp = save_path.split('_')[0]
        if yaml_file.export_yaml(save_time_stamp):
            print('Yaml file exported.')
        else:
            print('Yaml file export failed.')
    except Exception as e:
        print(str(e))
        with open('error_log.log', 'a') as log:
            error_msg = f'{stage} failed.'
            log_time = time.localtime(time.time())
            log.write(
                f'{log_time.tm_year}/{log_time.tm_mon}/{log_time.tm_mday} {log_time.tm_hour}:{log_time.tm_min}' + ':\n')
            log.write(f'Error: {error_msg}\n')
            log.write(f'Expectation: {e}\n')
            log.write('-' * 40 + '\n')
        exit(error_msg)