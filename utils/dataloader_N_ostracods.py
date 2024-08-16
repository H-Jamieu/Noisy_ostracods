from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
import torch

class ostracod_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=79): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            
        
        with open('./datasets/ostracods_genus_final_train.csv','r') as f:
            lines = f.read().splitlines()
            for l in lines:
                # reading "image_path, label"
                entry = l.split(',')          
                img_path = '%s/'%self.root+entry[0] # image path to absolute path
                self.train_labels[img_path] = int(entry[1].strip()) # label to integer                     
        with open('./datasets/ostracods_genus_clean_test.csv','r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split(',')           
                img_path = '%s/'%self.root+entry[0]
                self.test_labels[img_path] = int(entry[1])

        if mode == 'all':
            train_imgs=[]
            with open('./datasets/ostracods_genus_final_train.csv','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    # reading "image_path, label"
                    entry = l.split(',')          
                    img_path = '%s/'%self.root+entry[0] # image path to absolute path
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                #if class_num[label]<(num_samples/79) and len(self.train_imgs)<num_samples:
                if len(self.train_imgs)<num_samples: # this dataset cannot be balanced, so remove the class_num condition
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)       
        elif self.mode == "labeled":   
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            # filter out the images out of the dataset
            pred_idx = [i for i in pred_idx if i<len(train_imgs) and i>=0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
            print(f'the size difference is {len(pred.nonzero()[0]) - len(self.train_imgs)}')
            print(f'the size of the probability is {len(paths)}')
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]
            pred_idx = [i for i in pred_idx if i<len(train_imgs) and i>=0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs = []
            with open('./datasets/ostracods_genus_clean_test.csv','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    file_data = l.split(',')[0]
                    img_path = '%s/'%self.root+file_data
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('./datasets/ostracods_genus_clean_test.csv','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    file_data = l.split(',')[0]
                    img_path = '%s/'%self.root+file_data
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
class ostracod_dataloader():  
    def __init__(self, root, batch_size, num_batches, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
                   
        self.transform_train = transforms.Compose([transforms.Resize([224, 224]),
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
        self.transform_test = transforms.Compose([transforms.Resize([224, 224]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])        
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = ostracod_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = ostracod_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = ostracod_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = ostracod_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = ostracod_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=384,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             
        elif mode=='val':
            val_dataset = ostracod_dataset(self.root,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=384,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader     