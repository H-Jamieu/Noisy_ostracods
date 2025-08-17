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
import transformers
# https://github.com/facebookresearch/dinov3
# pip install -e .
# then you can import
# you need to provide the weights yourself per request of mata
from dinov3.hub.backbones import dinov3_vitl16

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

class DynamicResolutionDataset(Dataset):
    """
    Dataset for dynamic resolution models like DINOv3
    Returns images with their original aspect ratios preserved and patch-size aligned
    """
    
    def __init__(self, annotations_file, min_size=224, max_size=1024, preserve_aspect_ratio=True, 
                 patch_size=16, resize_method='nearest'):
        """
        :param annotations_file: List of image paths
        :param min_size: Minimum size for the shorter edge
        :param max_size: Maximum size for the longer edge
        :param preserve_aspect_ratio: Whether to preserve aspect ratio
        :param patch_size: Patch size for the model (e.g., 16 for ViT)
        :param resize_method: 'nearest' for nearest neighbor resizing, 'padding' for symmetric padding
        """
        self.imgs = annotations_file
        self.min_size = min_size
        self.max_size = max_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.patch_size = patch_size
        self.resize_method = resize_method
        
        # Cache image sizes for efficient batching
        self.image_sizes = []
        self._cache_image_sizes()
        
    def _make_divisible_by_patch_size(self, size):
        """
        Make a size divisible by patch_size using nearest neighbor approach
        :param size: Original size (width or height)
        :return: Size divisible by patch_size
        """
        remainder = size % self.patch_size
        if remainder == 0:
            return size
        
        # Find nearest divisible size
        lower = size - remainder
        upper = size + (self.patch_size - remainder)
        
        # Choose the nearest one
        if remainder <= self.patch_size // 2:
            return lower if lower > 0 else self.patch_size
        else:
            return upper
    
    def _calculate_symmetric_padding(self, size):
        """
        Calculate symmetric padding to make size divisible by patch_size
        :param size: Original size (width or height)
        :return: (pad_left/top, pad_right/bottom)
        """
        remainder = size % self.patch_size
        if remainder == 0:
            return 0, 0
        
        total_padding = self.patch_size - remainder
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        
        return pad_left, pad_right
        
    def _cache_image_sizes(self):
        """Cache image sizes to enable size-based batching"""
        print("Caching image sizes for efficient batching...")
        for img_path in tqdm(self.imgs):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    # Calculate the target size after transform
                    if self.preserve_aspect_ratio:
                        scale = self.min_size / min(w, h)
                        if max(w, h) * scale > self.max_size:
                            scale = self.max_size / max(w, h)
                        target_w, target_h = int(w * scale), int(h * scale)
                    else:
                        target_w, target_h = self.min_size, self.min_size
                    
                    # Make sizes divisible by patch_size
                    if self.resize_method == 'nearest':
                        target_w = self._make_divisible_by_patch_size(target_w)
                        target_h = self._make_divisible_by_patch_size(target_h)
                    elif self.resize_method == 'padding':
                        # For padding, we don't change target size here, padding is added later
                        pass
                    
                    self.image_sizes.append((target_w, target_h))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                # Default to min_size aligned to patch_size
                aligned_size = self._make_divisible_by_patch_size(self.min_size)
                self.image_sizes.append((aligned_size, aligned_size))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Dynamic resize while preserving aspect ratio
        w, h = image.size
        if self.preserve_aspect_ratio:
            scale = self.min_size / min(w, h)
            if max(w, h) * scale > self.max_size:
                scale = self.max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = self.min_size, self.min_size
        
        # Apply patch size alignment strategy
        if self.resize_method == 'nearest':
            # Method 1: Resize to nearest patch-size divisible dimensions
            aligned_w = self._make_divisible_by_patch_size(new_w)
            aligned_h = self._make_divisible_by_patch_size(new_h)
            image = image.resize((aligned_w, aligned_h), Image.LANCZOS)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return transform(image), img_path
            
        elif self.resize_method == 'padding':
            # Method 2: Resize to target size then add symmetric padding
            image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Convert to tensor first
            tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = tensor_transform(image)
            
            # Calculate padding for each dimension
            pad_left, pad_right = self._calculate_symmetric_padding(new_w)
            pad_top, pad_bottom = self._calculate_symmetric_padding(new_h)
            
            # Apply padding (format: [pad_left, pad_right, pad_top, pad_bottom])
            image_tensor = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            
            return image_tensor, img_path
        
        else:
            raise ValueError(f"Unknown resize_method: {self.resize_method}. Use 'nearest' or 'padding'.")

def dynamic_collate_fn(batch):
    """
    Custom collate function for dynamic resolution batching
    Groups images by similar sizes to enable efficient batching
    """
    # Sort by image size (area) to group similar sizes together
    batch = sorted(batch, key=lambda x: x[0].shape[1] * x[0].shape[2])
    
    images, paths = zip(*batch)
    
    # Try to create batches of similar-sized images
    batched_images = []
    batched_paths = []
    
    i = 0
    while i < len(images):
        current_batch_images = [images[i]]
        current_batch_paths = [paths[i]]
        current_size = (images[i].shape[1], images[i].shape[2])
        
        # Look for images of the same size
        j = i + 1
        while j < len(images) and len(current_batch_images) < 8:  # Max batch size of 8
            if images[j].shape[1] == current_size[0] and images[j].shape[2] == current_size[1]:
                current_batch_images.append(images[j])
                current_batch_paths.append(paths[j])
                j += 1
            else:
                break
        
        if len(current_batch_images) > 1:
            # Stack images of the same size
            batched_images.append(torch.stack(current_batch_images))
            batched_paths.append(current_batch_paths)
        else:
            # Single image batch
            batched_images.append(current_batch_images[0].unsqueeze(0))
            batched_paths.append(current_batch_paths)
        
        i = j if j > i + 1 else i + 1
    
    return batched_images, batched_paths

def single_image_collate_fn(batch):
    """
    Simple collate function that processes one image at a time
    Best for maximum compatibility with dynamic resolution models
    """
    images, paths = zip(*batch)
    return [(img.unsqueeze(0), [path]) for img, path in zip(images, paths)]

def get_image_embedding(model, device, input_loader, scaler=True):
    model_for_embedding = model
    # remove projection layer to get the image features without being projected to the shared space
    #model_for_embedding.proj = None
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
                # don't need mask and normalization
                embedding, mask, id_restore = model.forward_encoder(inputs, mask_ratio=0)
                # remove the cls_token at the begining of the embedding
                embedding = embedding[:, 1:, :]
                # average pooling the embeddings
                embedding = embedding.mean(dim=1)
                embedding = embedding.view(embedding.size(0), -1)
                image_embedding = torch.cat((image_embedding, embedding), 0)
                iter_counter += inputs.size(0)
                del embedding, inputs
                
                # Offload to CPU when reaching threshold
                if iter_counter >= 24000:
                    offloaded_embedding.append(image_embedding.cpu())
                    image_embedding = torch.tensor(()).to(device)
                    torch.cuda.empty_cache()
                    iter_counter = 0
    
    # merge the offloaded embedding
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
                # don't need mask and normalization
                embedding = model(inputs)
                # embedding is shape of (batch_size, 1536), just stacking is fine
                image_embedding = torch.cat((image_embedding, embedding), 0)
                iter_counter += inputs.size(0)
                del embedding, inputs
                
                if iter_counter >= 24000:
                    offloaded_embedding.append(image_embedding.cpu())
                    image_embedding = torch.tensor(()).to(device)
                    torch.cuda.empty_cache()
                    iter_counter = 0
    # merge the offloaded embedding
    if len(offloaded_embedding) > 0:
        offloaded_embedding.append(image_embedding.cpu())
        image_embedding = torch.cat(offloaded_embedding, 0)
    return image_embedding

def get_image_embedding_dinov3_dynamic(model, device, input_loader, scaler=True):
    """
    Extract embeddings from DINOv3 with dynamic resolution support
    Handles variable-sized batches efficiently
    """
    all_embeddings = []
    all_paths = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=scaler is not None):
            for batch_data in tqdm(input_loader):
                # Handle different collate function outputs
                if isinstance(batch_data, list) and len(batch_data) > 0 and isinstance(batch_data[0], tuple):
                    # single_image_collate_fn output: list of (tensor, [path])
                    for inputs, paths in batch_data:
                        inputs = inputs.to(device)
                        embedding = model(inputs)
                        all_embeddings.append(embedding.cpu())
                        all_paths.extend(paths)
                        del embedding, inputs
                else:
                    # dynamic_collate_fn output: (batched_images, batched_paths)
                    batched_images, batched_paths = batch_data
                    for images, paths in zip(batched_images, batched_paths):
                        images = images.to(device)
                        embedding = model(images)
                        all_embeddings.append(embedding.cpu())
                        all_paths.extend(paths)
                        del embedding, images
                
                # Periodic cleanup
                if len(all_embeddings) % 100 == 0:
                    torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    return final_embeddings, all_paths

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

def extract_clip_embeddings():
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #chkpt_dir = '/mnt/d/working_dir/mae/output_dir/mae_visualize_vit_large.pth' # customize this to your model checkpoint
    #chkpt_dir = 'd:/working_dir/mae/output_dir/mae_vit_large_patch16_1715962570.925112.pth'
    chkpt_dir = '/mnt/c/users/windowsshit/working_dir/mae/output_dir/mae_vit_huge_patch14_1744809191.3047493.pth'
    img_dir = "/mnt/e/play_ground/training_data"
    pt = torch.load(chkpt_dir, map_location=device)
    #print(pt.keys())
    default_weight_keys = ['mae_vit_large_path16', 'mae_vit_base_patch16', 'mae_vit_huge_patch14']
    # find the keys in the chkpt_path
    the_key = 'mae_vit_large_path16'
    for k in default_weight_keys:
        if k in chkpt_dir:
            the_key = k
            break
    model = models_mae.__dict__[the_key](norm_pix_loss=True) # customize this to your model
    model.load_state_dict(pt)
    # customize this to your image directory
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

def extract_DINOv3_embeddings(resize_method = 'nearest'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the model instance first, then load weights
    # For ViT-L weights, we should use a ViT-L model
    try:
        model = dinov3_vitl16(pretrained=False)  # Don't load pretrained weights yet
        
        # Load the local weights
        vit_l16_local_path = "D:/Explores/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        state_dict = torch.load(vit_l16_local_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print("DINOv3 model loaded successfully!")
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}")
        return

    model.eval()
    model.to(device)
    
    img_dir = "e:/ostracods_id/pseudo_annotation/ostracods"  # customize this to your image directory
    image_list = list_all_images(img_dir)
    
    print(f"Found {len(image_list)} images")
    
    # Configuration options
    use_dynamic_batching = True  # Set to False for single-image processing
    patch_size = 16              # ViT patch size (typically 16 or 14)
        # 'nearest' or 'padding'
    
    print(f"Using patch size: {patch_size}")
    print(f"Using resize method: {resize_method}")
    
    if use_dynamic_batching:
        print("Using dynamic resolution with smart batching...")
        dataset = DynamicResolutionDataset(
            image_list, 
            min_size=224,                    # Minimum size for shorter edge
            max_size=896,                    # Maximum size for longer edge  
            preserve_aspect_ratio=True,
            patch_size=patch_size,           # Make sizes divisible by patch size
            resize_method=resize_method      # Choose resize strategy
        )
        # Use custom collate function for batching similar-sized images
        dataloader = DataLoader(
            dataset, 
            batch_size=16,    # This will be dynamically adjusted
            shuffle=False, 
            num_workers=8,
            collate_fn=dynamic_collate_fn
        )
        image_embedding, paths = get_image_embedding_dinov3_dynamic(model, device, dataloader)
        # Ensure paths match the original order
        path_to_embedding = {path: emb for path, emb in zip(paths, image_embedding)}
        ordered_embeddings = [path_to_embedding[path] for path in image_list]
        image_embedding = torch.stack(ordered_embeddings)
    else:
        print("Using single-image processing for maximum compatibility...")
        dataset = DynamicResolutionDataset(
            image_list,
            min_size=224,
            max_size=896,
            preserve_aspect_ratio=True,
            patch_size=patch_size,
            resize_method=resize_method
        )
        # Process one image at a time
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=single_image_collate_fn
        )
        image_embedding, paths = get_image_embedding_dinov3_dynamic(model, device, dataloader)
    
    del model
    image_embedding = image_embedding.cpu().numpy()
    torch.cuda.empty_cache()
    
    output_file = f"../datasets/embeddings/image_embeddings_490k_dynamic_DINOv3_vitl16_{resize_method}_p{patch_size}.csv"
    save_to_csv(image_embedding, output_file, image_list)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    #extract_clip_embeddings()
    #extract_mae_embeddings()
    #extract_DINO_embeddings()
    extract_DINOv3_embeddings(resize_method='nearest')
    extract_DINOv3_embeddings(resize_method='padding')