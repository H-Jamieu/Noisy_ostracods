import os
import torch
from tqdm import tqdm
from trainer import CustomImageDataset, determine_model, modify_model, determine_device
import utils.customizedYaml as customizedYaml
from torchvision.transforms import v2 as transforms

def batched_inference(input_loader, model, device, scaler=None):
    """
    input: the processed normalized image, made into batch or not, not shuffled
    model: the model for inference image
    return: all predictions to the data.
    """
    model.eval()
    model.to(device)
    all_predictions = torch.tensor(()).to(device)
    batch_sz = input_loader.batch_size
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            for images, labels, idx in tqdm(input_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                all_predictions = torch.cat((all_predictions, predictions), 0)
    return all_predictions

def cincer(input_loader, model, device, scaler=None):
    """
    input: the processed normalized image, made into batch or not, not shuffled
    model: the model for inference image
    return: all predictions to the data.
    """
    model.eval()
    model.to(device)
    all_predictions = torch.tensor(()).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            for images, labels, idx in tqdm(input_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # convert the predictions to probability
                outputs= torch.nn.functional.softmax(outputs, dim=1)
                # Find the top prediction
                top_prob, top_pred = torch.max(outputs, dim=1)
                # Calculate the margin as the difference between top prediction and annotated label
                annot_prob = outputs[torch.arange(outputs.size(0)), labels]
                margin = top_prob - annot_prob
                # Save the margin into all_predictions
                all_predictions = torch.cat((all_predictions, margin), 0)
    return all_predictions

def calculate_metrics(predictions, labels, output_path=None):
    """
    predictions: the predictions from the model
    labels: the ground truth labels
    return: the metrics for the model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return accuracy, precision, recall, f1

def cincer_filter(predictions, all_imgs, tau, output_path='cincer_filtered.csv'):
    """
    Find the images having the margin smaller than tau
    """
    # convert predictions to np array
    predictions = predictions.cpu().numpy()
    # geting the indexes with margin smaller than tau
    small_idx = predictions > tau
    # print pidenction statistics
    mean = predictions.mean()
    std = predictions.std()
    print(f'Mean: {mean}, Std: {std}')
    filtered_imgs = all_imgs[small_idx]
    if output_path is not None:
        filtered_imgs.to_csv(output_path, index=False, header=False)
    return filtered_imgs

def evaluate_model(config_path, batch_size=32, output=False, task='classification'):
    """
    config_path: the path to the config file
    model_path: the path to the model
    data_path: the path to the data
    batch_size: the batch size for inference
    """
    yaml_file = customizedYaml.yaml_handler(config_path)
    args = yaml_file.data
    device, _ = determine_device(args['device'])
    model = determine_model(args['model'], '')
    model_path = args['model_path']
    modify_model(args['model'], model, args['classes'])
    model.load_state_dict(torch.load(model_path))
    guidence_file_path = './datasets/'+args['target']+'_test.csv'
    data_path = args['base_path']+'/'+args['class_img_path']
    print(f'Evaluating model {model_path} on {guidence_file_path}')
    valid_test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    dataset = CustomImageDataset(guidence_file_path, data_path, valid_test_transforms)
    labels = dataset.img_labels[1].values
    input_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if task == 'cincer':
        guidence_file_path_val = './datasets/'+args['target']+'_val.csv'
        dataset_val = CustomImageDataset(guidence_file_path_val, data_path, valid_test_transforms)
        input_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        predictions = cincer(input_loader, model, device)
        predictions_val = cincer(input_loader_val, model, device)
        cincer_filter(predictions, dataset.img_labels, 0.2)
        cincer_filter(predictions_val, dataset_val.img_labels, 0.2, 'cincer_filtered_val.csv')
        return (1, 1, 1, 1)
    predictions = batched_inference(input_loader, model, device)
    predictions = predictions.cpu().numpy().astype(int)
    metrics = calculate_metrics(predictions, labels)
    if output:
        # output: image, label, prediction
        output_path = './log_dir/' + args['model_path'].split('/')[-1].split('.pth')[0] + '_predictions.csv'
        out_file = dataset.img_labels.copy()
        out_file[2] = predictions
        out_file.to_csv(output_path, index=False, header=False)
    return metrics

if __name__ == '__main__':
    config_path = 'config_evaluation.yaml'
    data_path = './datasets'
    metrics = evaluate_model(config_path, 128, True, 'cincer')
    print(f'{metrics[0]} {metrics[1]} {metrics[2]} {metrics[3]}')