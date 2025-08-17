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

# Import necessary classes for SigLIP 2 and DINOv2
import transformers
from transformers import SiglipModel, AutoProcessor

"""
Extract the features from the images using the CLIP, MAE, DINOv2, and SigLIP 2 models
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
        :param transform: Transformation applied to images. Should be a torchvision.transform type.
        :param image_h, image_w: the weight and height for inference
        """
        self.imgs = annotations_file
        self.transform = transform
        self.image_w, self.image_h = image_w, image_h
        # If a specific transform is not provided, build a default one.
        if self.transform is None:
            self.build_transforms()

    def __len__(self):
        return len(self.imgs)

    def __classes__(self):
        labels = set([x[1] for x in self.imgs])
        return len(labels)

    def build_transforms(self):
        # This default transform is used if a custom one isn't passed during initialization.
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
        """
        img_path = self.imgs[idx]
        # Ensure image is in RGB format
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image

def get_image_embedding(model, device, input_loader, scaler=True):
    model_for_embedding = model
    model_for_embedding.eval()
    model_for_embedding.to(device)
    image_embedding = torch.tensor(()).to(device)
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
    iter_counter = 0
    offloaded_embedding = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            for inputs in tqdm(input_loader):
                inputs = inputs.to(device)
                embedding, mask, id_restore = model.forward_encoder(inputs, mask_ratio=0)
                embedding = embedding[:, 1:, :]
                embedding = embedding.mean(dim=1)
                embedding = embedding.view(embedding.size(0), -1)
                image_embedding = torch.cat((image_embedding, embedding), 0)
                iter_counter += inputs.size(0)
                del embedding, inputs
                
                if iter_counter >= 24000:
                    offloaded_embedding.append(image_embedding.cpu())
                    image_embedding = torch.tensor(()).to(device)
                    torch.cuda.empty_cache()
                    iter_counter = 0
    
    if len(offloaded_embedding) > 0:
        offloaded_embedding.append(image_embedding.cpu())
        image_embedding = torch.cat(offloaded_embedding, 0)
    return image_embedding

def get_image_embedding_dino(model, device, input_loader, scaler=True):
    image_embedding = torch.tensor(()).to(device)
    model.eval()
    model.to(device)
    iter_counter = 0
    offloaded_embedding = []
    with torch.no_grad():
        with torch.autocast(device_type = 'cuda', dtype=torch.float16, enabled=scaler is not None):
            for inputs in tqdm(input_loader):
                inputs = inputs.to(device)
                embedding = model(inputs)
                image_embedding = torch.cat((image_embedding, embedding), 0)
                iter_counter += inputs.size(0)
                del embedding, inputs
                
                if iter_counter >= 24000:
                    offloaded_embedding.append(image_embedding.cpu())
                    image_embedding = torch.tensor(()).to(device)
                    torch.cuda.empty_cache()
                    iter_counter = 0

    if len(offloaded_embedding) > 0:
        offloaded_embedding.append(image_embedding.cpu())
        image_embedding = torch.cat(offloaded_embedding, 0)
    return image_embedding

# --- NEW FUNCTION FOR SIGLIP 2 ---
def get_image_embedding_siglip(model, device, input_loader, scaler=True):
    """
    Extracts image embeddings using a SigLIP model.
    """
    image_embedding = torch.tensor(()).to(device)
    model.eval()
    # The model is already on the correct device(s) if using device_map='auto'
    
    iter_counter = 0
    offloaded_embedding = []
    
    with torch.no_grad():
        # SigLIP models are typically used with float16 for efficiency
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(scaler is not None and device == 'cuda')):
            for inputs in tqdm(input_loader, desc="Extracting SigLIP Embeddings"):
                inputs = inputs.to(device)
                
                # Use the get_image_features method to extract embeddings
                embedding = model.get_image_features(pixel_values=inputs)
                
                image_embedding = torch.cat((image_embedding, embedding), 0)
                iter_counter += inputs.size(0)
                del embedding, inputs
                
                # Offload to CPU when reaching threshold (same logic as your other functions)
                if iter_counter >= 24000:
                    offloaded_embedding.append(image_embedding.cpu())
                    image_embedding = torch.tensor(()).to(device)
                    torch.cuda.empty_cache()
                    iter_counter = 0
    
    # Merge embeddings offloaded to CPU
    if len(offloaded_embedding) > 0:
        offloaded_embedding.append(image_embedding.cpu())
        image_embedding = torch.cat(offloaded_embedding, 0)
        
    return image_embedding

def save_to_csv(embeddings, file_name, image_paths):
    """
    Save the embeddings to a csv file:
        format: image_path, embedding
    """
    with open(file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        for i in tqdm(range(len(embeddings)), desc=f"Saving to {file_name}"):
            embedding_element = embeddings[i].tolist()
            embedding_element.insert(0, image_paths[i])
            csv_writer.writerow(embedding_element)

def ectract_clip_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    model.to(device)
    img_dir = "ostracods_data/class_images" # customize this to your image directory
    image_list = list_all_images(img_dir)
    # Note: For best results, the `preprocess` object from CLIP should be used.
    dataset = CompactDataset(image_list, transform=preprocess, image_w=336, image_h=336)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    image_embedding = get_image_embedding(model, device, dataloader)
    del model
    image_embedding = image_embedding.cpu().numpy()
    torch.cuda.empty_cache()
    save_to_csv(image_embedding, "image_embeddings.csv", image_list)

def extract_mae_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chkpt_dir = '/mnt/c/users/windowsshit/working_dir/mae/output_dir/mae_vit_huge_patch14_1744809191.3047493.pth'
    img_dir = "/mnt/e/play_ground/training_data"
    pt = torch.load(chkpt_dir, map_location=device)
    default_weight_keys = ['mae_vit_large_path16', 'mae_vit_base_patch16', 'mae_vit_huge_patch14']
    the_key = 'mae_vit_large_path16'
    for k in default_weight_keys:
        if k in chkpt_dir:
            the_key = k
            break
    model = models_mae.__dict__[the_key](norm_pix_loss=True)
    model.load_state_dict(pt)
    image_list = list_all_images(img_dir)
    customized_transform = transforms.Compose([
                                                 transforms.Resize([224, 224]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.26527297, 0.26252866, 0.25425115],
                                                     std=[0.24846438, 0.24605624, 0.236864])])
    dataset = CompactDataset(image_list, transform=customized_transform, image_w=224, image_h=224)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)
    image_embedding = get_image_embedding_mae(model, device, dataloader)
    del model
    image_embedding = image_embedding.cpu().numpy()
    torch.cuda.empty_cache()
    save_to_csv(image_embedding, "../datasets/embeddings/image_embeddings_470k_224_mae_full_pretrain.csv", image_list)

def extract_DINO_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_v2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dino_v2_model.eval()
    dino_v2_model.to(device)
    img_dir = "/mnt/e/play_ground/training_data"
    image_list = list_all_images(img_dir)
    dataset = CompactDataset(image_list, transform=None, image_w=224, image_h=224)
    dataloader = DataLoader(dataset, batch_size=48, shuffle=False, num_workers=16)
    image_embedding = get_image_embedding_dino(dino_v2_model, device, dataloader)
    del dino_v2_model
    image_embedding = image_embedding.cpu().numpy()
    torch.cuda.empty_cache()
    save_to_csv(image_embedding, "../datasets/embeddings/image_embeddings_470k_224_DINOv2_g14_full.csv", image_list)

# --- NEW FUNCTION TO EXTRACT SIGLIP 2 EMBEDDINGS ---
def extract_siglip_embeddings():
    """
    Orchestrates the extraction of image embeddings using the SigLIP 2 model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/siglip2-so400m-patch14-384"
    
    # Load the SigLIP model and processor
    # Using float16 and device_map='auto' for efficient loading on available hardware (GPU/CPU)
    model = SiglipModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # --- Critical Step: Create the correct transform for SigLIP ---
    # We get the required image size, mean, and std from the processor
    # to ensure the input images are preprocessed exactly as the model expects.
    image_processor = processor.image_processor
    image_size = 384 # Should be 384 for this model
    
    siglip_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    # --- The rest of the logic follows your established pattern ---
    img_dir = "e:/ostracods_id/class_images" # <<< CUSTOMIZE THIS to your image directory
    image_list = list_all_images(img_dir)
    
    # Initialize the dataset with the specific SigLIP transform
    dataset = CompactDataset(image_list, transform=siglip_transform, image_w=image_size, image_h=image_size)
    
    # Adjust batch_size based on your GPU memory
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # Call the new function to get embeddings
    image_embedding = get_image_embedding_siglip(model, device, dataloader)
    
    del model
    image_embedding = image_embedding.cpu().numpy()
    torch.cuda.empty_cache()
    
    # Save the results to a new CSV file
    output_path = "../datasets/embeddings/image_embeddings_siglip2_so400m_p14_384.csv" # <<< CUSTOMIZE THIS output path
    save_to_csv(image_embedding, output_path, image_list)


if __name__ == "__main__":
    #ectract_clip_embeddings()
    #extract_mae_embeddings()
    #extract_DINO_embeddings()
    
    # Run the new SigLIP 2 embedding extraction
    extract_siglip_embeddings()