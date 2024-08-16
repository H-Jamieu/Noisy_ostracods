import torch
import clip
import csv
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import models_mae
"""
Extract the features from the images using the CLIP and MAE model
"""

def list_all_images(directory):
    """
    Lists all images in a directory
    :param directory: The directory to list images from
    :return: List of images in the directory
    """
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                images.append(os.path.join(root, file))
    # remove all negative class images
    images = [x for x in images if "negative" not in x]
    return images

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class CompactDataset(Dataset):
    """
    DataLoader class. Sub-class of torch.utils.data Dataset class. It will load data from designated files.
    """

    def __init__(self, annotations_file, transform=None, image_w=224, image_h=224):
        """
        Initial function. Creates the instance.
        :param annotations_file: The file containing image directory and labels (for train and validation)
        :param img_dir: The directory containing target images.
        :param transform: Transformation applied to images. Should be a torchvision.transform type.
        :param image_h, image_w: the weight and height for inference
        """
        self.imgs = annotations_file
        self.transform = transform
        self.image_w, self.image_h = image_w, image_h
        self.build_transforms(transform)

    def __len__(self):
        return len(self.imgs)

    def __classes__(self):
        labels = set([x[1] for x in self.imgs])
        return len(labels)

    def build_transforms(self, transform):
        if transform is None:
            self.transform = transforms.Compose([#SquarePad(),
                                                 transforms.Resize([self.image_w, self.image_w]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        """
        Controls what returned from data loader
        :param idx: Index of image.
        :return: image: The image array.
        label: label of training images.
        img_path: path to the image.
        """
        img_path = self.imgs[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image

def get_image_embedding(model, device, input_loader, scaler=True):
    model_for_embedding = model
    # remove projection layer to get the image features without being projected to the shared space
    #model_for_embedding.proj = None
    model_for_embedding.eval()
    model_for_embedding.to(device)
    image_embedding = torch.tensor(()).to(device)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            for inputs in tqdm(input_loader):
                inputs = inputs.to(device)
                out = model_for_embedding.encode_image(inputs)
                image_embedding = torch.cat((image_embedding, out), 0)
    return image_embedding

def get_image_embedding_mae(model, device, input_loader, scaler=True):
    image_embedding = torch.tensor(()).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            for inputs in tqdm(input_loader):
                inputs = inputs.to(device)
                # don't need mask and normalization
                embedding, mask, id_restore = model.forward_encoder(inputs, mask_ratio=0)
                # remove the cls_token at the begining of the embedding
                embedding = embedding[:, 1:, :]
                # average pooling the embeddings
                embedding = embedding.mean(dim=1)
                embedding = embedding.view(embedding.size(0), -1)
                image_embedding = torch.cat((image_embedding, embedding), 0)
                del embedding, inputs
    return image_embedding

def save_to_csv(embeddings, file_name, image_paths):
    """
    Save the embeddings to a csv file:
        format: image_path, embedding
    """
    with open(file_name, 'w') as f:
        csv_writer = csv.writer(f)
        for i in tqdm(range(len(embeddings))):
            embedding_element = embeddings[i].tolist()
            # add image path to the embedding by inserting it at the beginning of the list
            embedding_element.insert(0, image_paths[i])
            #line = [image_paths[i], embeddings[i][0]]
            csv_writer.writerow(embedding_element)

def ectract_clip_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    model.to(device)
    img_dir = "ostracods_data/class_images" # customize this to your image directory
    image_list = list_all_images(img_dir)
    dataset = CompactDataset(image_list, transform=None, image_w=336, image_h=336)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    image_embedding = get_image_embedding(model, device, dataloader)
    del model
    image_embedding = image_embedding.cpu().numpy()
    # release cuda memory
    torch.cuda.empty_cache()
    save_to_csv(image_embedding, "image_embeddings.csv", image_list)

def extract_mae_embeddings():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    chkpt_dir = '/mnt/d/working_dir/mae/output_dir/mae_visualize_vit_large.pth' # customize this to your model checkpoint
    #chkpt_dir = 'd:/working_dir/mae/output_dir/mae_vit_large_patch16_1715962570.925112.pth'
    pt = torch.load(chkpt_dir, map_location=device)
    model = models_mae.__dict__['mae_vit_large_patch16'](norm_pix_loss=True) # customize this to your model
    model.load_state_dict(pt['model'])
    img_dir = "/mnt/x/class_images" # customize this to your image directory
    #img_dir = "d:/ostracods_data/class_images"
    image_list = list_all_images(img_dir)
    # (0.25425115, 0.26252866, 0.26527297, 0.236864, 0.24605624, 0.24846438)
    # calculated the mean and std of the dataset (b,g,r)
    customized_transform = transforms.Compose([#SquarePad(),
                                                 transforms.Resize([224, 224]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.26527297, 0.26252866, 0.25425115],
                                                     std=[0.24846438, 0.24605624, 0.236864])])
    dataset = CompactDataset(image_list, transform=None, image_w=224, image_h=224)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)
    image_embedding = get_image_embedding_mae(model, device, dataloader)
    del model
    image_embedding = image_embedding.cpu().numpy()
    # release cuda memory
    torch.cuda.empty_cache()
    save_to_csv(image_embedding, "../datasets/embeddings/image_embeddings_mae_raw.csv", image_list)

if __name__ == "__main__":
    #ectract_clip_embeddings()
    extract_mae_embeddings()