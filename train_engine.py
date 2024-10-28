import torch
import time
import numpy as np
from tqdm import tqdm
import gc
from collections import deque
import higher

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k)
        return res

class base_meta_network(torch.nn.Module):
    """
    A base metanet for margin genration. The margin is class depedent
    Input: predict class index
    Output: margin for each class
    Source: https://github.com/jiangwenj02/dynamic_loss/blob/0863227d9b366e1979a65f1e08966484250f6e0e/mmcls/models/meta_net/dynamic_loss.py
    """

    def __init__(self, num_classes, h_dim=256):
        super(base_meta_network, self).__init__()
        self.num_classes = num_classes
        self.h_dim = h_dim
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self):
        self.cls_feature_emb = torch.nn.Parameter(torch.randn(1, self.h_dim))

        self.margin_generator = torch.nn.Sequential(
                                    torch.nn.Linear(self.h_dim, self.h_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.h_dim, self.h_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.h_dim, self.num_classes, bias=True) 
                                )
        
    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.cls_feature_emb)
        torch.nn.init.xavier_normal_(self.margin_generator[0].weight)
        torch.nn.init.xavier_normal_(self.margin_generator[2].weight)
        torch.nn.init.xavier_normal_(self.margin_generator[4].weight)
        self.margin_generator[0].bias.data.zero_()
        self.margin_generator[2].bias.data.zero_()
        # initialize the bias to 1 for preventing the meta train invalid
        torch.nn.init.constant_(self.margin_generator[4].bias, 1)

    def forward(self):
        margin = self.margin_generator(self.cls_feature_emb)
        return margin

    
def save_catached_tensor(catched_idx, save_path):
    """
       save as tensor easier to load for next next analysis
    """
    torch.save(catched_idx, save_path)
    print('Catch index saved at: ', save_path)

def loss_coteaching(y_1, y_2, t, forget_rate, idx):
    """
    adapted from https://github.com/xingruiyu/coteaching_plus/issues/1
    """
    loss_1 = torch.nn.functional.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = torch.nn.functional.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

    # getting the filtered samples
    ind_1_filtered = ind_1_sorted[num_remember:]
    ind_2_filtered = ind_2_sorted[num_remember:]
    idx_1_filtered = idx[ind_1_filtered]
    idx_2_filtered = idx[ind_2_filtered]
    
    # exchange
    loss_1_update = torch.nn.functional.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = torch.nn.functional.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update), torch.sum(loss_2_update), idx_1_filtered, idx_2_filtered

def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step):
    """
    adapted from https://github.com/xingruiyu/coteaching_plus/issues/1
    Fixing known bugs in the original code
    Also, recommend the batch size being as large as possible, as the forget_rate is usually very small
    """

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            # getting the index of the samples that two networks disagree with each other
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind.cpu().numpy()*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        # some times disagree_id is longer than ind_disagree?
        disagree_id = disagree_id[:ind_disagree.shape[0]]
    
    # what does this do? If filter out the loss with all aggremment after 5000 images, seems very questionable
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = torch.autograd.Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        # has disagreement on samples
        # when samples has disagreement, we only update the loss on the samples that has disagreement
        # very very questionable, as disagreement is usually small within a mini-batch, which will lead to a very small update
        update_labels = labels[disagree_id]
        update_outputs = logits[disagree_id] 
        update_outputs2 = logits2[disagree_id]
        # make ind_disagree as tensor, move to GPU
        ind_disagree = torch.tensor(ind_disagree, dtype=torch.int64).cuda()
        
        loss_1, loss_2, idx_1_filtered, idx_2_filtered = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree)
    else:
        update_labels = labels
        update_outputs = logits
        update_outputs2 = logits2

        # Original code, could be wrong
        # cross_entropy_1 = torch.nn.functional.cross_entropy(update_outputs, update_labels)
        # cross_entropy_2 = torch.nn.functional.cross_entropy(update_outputs2, update_labels)

        cross_entropy_1 = torch.nn.functional.cross_entropy(update_outputs, update_labels, reduction='none')
        cross_entropy_2 = torch.nn.functional.cross_entropy(update_outputs2, update_labels, reduction='none')

        # if step > 5000, zero out loss
        # Super questionable, seems only work for small datasets such as CIFAR10
        # one more problem is that nn.CrossEntropyLoss() has already averaged the loss, so the loss is already divided by the batch size
        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]

        idx_1_filtered = torch.tensor([], dtype=torch.int64).cuda()
        idx_2_filtered = torch.tensor([], dtype=torch.int64).cuda()
    return loss_1, loss_2, idx_1_filtered, idx_2_filtered

def loss_mentor(y_1, t, forget_rate, idx):
    """
    adapted from 
    """
    loss_1 = torch.nn.functional.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]

    # getting the filtered samples
    ind_1_filtered = ind_1_sorted[num_remember:]
    idx_1_filtered = idx[ind_1_filtered]

    # filter
    loss_1_update = torch.nn.functional.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    
    return loss_1_update, idx_1_filtered

class loss_transition_matrix(torch.nn.Module):
    def __init__(self, transition_matrix):
        super(loss_transition_matrix, self).__init__()
        self.transition_matrix = transition_matrix

    def forward(self, y, t):
        """
        adding adaption loss to the model with known transition matrix
        Original work: Sukhbaatar et al., 2015, equation 3
        """
        # if len(y) != len(transition_matrix):
        #     # in case the last batch is smaller than the transition matrix
        #     transition_matrix = transition_matrix[0].repeat(len(y), 1, 1)
        # create a diginal matrix of size transition_matrix
        # casting inputs to fp32

        dig_matrix = self.transition_matrix #torch.eye(self.transition_matrix.size(0)).cuda() #

        with torch.cuda.amp.autocast(enabled=False):
            y_fp32 = y.float()
            transition_matrix_fp32 = dig_matrix
            
            batch_probs = torch.nn.functional.softmax(y_fp32, dim=1)
            adjusted_probs = torch.matmul(batch_probs, transition_matrix_fp32)
            log_modified_outputs = torch.log(adjusted_probs + 1e-12)
            loss = torch.nn.functional.nll_loss(log_modified_outputs, t)
            
            # Cast back to original dtype
            loss = loss.to(y.dtype)
        return loss

class EfficientAUMCalculator:
    def __init__(self, num_samples, device):
        self.aum_values = torch.zeros(num_samples, dtype=torch.float32, device=device)
        self.update_counts = torch.zeros(num_samples, dtype=torch.int32, device=device)
        self.device = device

    def update(self, logits, targets, sample_ids):
        with torch.no_grad():
            batch_size, num_classes = logits.shape

            # Ensure sample_ids is on the correct device
            sample_ids = sample_ids.to(self.device)

            # Get the target logits, logits[batch_idx, class_GT]
            target_logits = logits[torch.arange(batch_size, device=self.device), targets]

            # Create a mask to exclude the target logits
            mask = torch.ones_like(logits)
            mask[torch.arange(batch_size, device=self.device), targets] = 0

            # Get the highest non-target logits
            max_non_target_logits = (logits * mask).max(dim=1).values

            # Calculate margin M = target_logit - max(non_target_logits)
            margins = target_logits - max_non_target_logits

            # convert margins to FP32
            margins = margins.to(torch.float32)

            # Update AUM values and counts
            self.aum_values.index_add_(0, sample_ids, margins)
            self.update_counts.index_add_(0, sample_ids, torch.ones_like(sample_ids, dtype=torch.int32, device=self.device))

    def get_aum_ranking(self):
        with torch.no_grad():
            # Avoid division by zero
            valid_counts = torch.clamp(self.update_counts, min=1)
            average_aum = self.aum_values / valid_counts
            return average_aum.cpu().numpy()

    def save_aum_ranking(self, filepath):
        aum_ranking = self.get_aum_ranking()
        torch.save(aum_ranking, filepath)

# def loss_transistion_matrix(y, t, transition_matrix):
#     """
#     adding adaption loss to the model with known transition matrix
#     Original work: Sukhbaatar et al., 2015, equation 3
#     """
#     # if len(y) != len(transition_matrix):
#     #     # in case the last batch is smaller than the transition matrix
#     #     transition_matrix = transition_matrix[0].repeat(len(y), 1, 1)
#     # create a diginal matrix of size transition_matrix
#     dig_matrix = torch.eye(transition_matrix.size(0)).cuda()

#     batch_probs = torch.nn.functional.softmax(y, dim=1)
#     #batch_probs = batch_probs.unsqueeze(-1)
#     #adjusted_probs = torch.matmul(batch_probs, transition_matrix)
#     adjusted_probs = torch.matmul(batch_probs, dig_matrix)
#     #adjusted_probs = adjusted_probs.squeeze(-1)
#     log_modified_outputs = torch.log(adjusted_probs)
#     loss = torch.nn.functional.nll_loss(log_modified_outputs, t)
#     return loss

class Loss_Recorder:
    '''
    Record the latest loss for each image
    '''
    def __init__(self, num_samples, device):
        self.loss_values = torch.zeros(num_samples, dtype=torch.float32, device=device)
        self.device = device
    
    def update(self, loss, sample_ids):
        with torch.no_grad():
            sample_ids = sample_ids.to(self.device)
            self.loss_values.index_copy_(0, sample_ids, loss)

    def get_loss_by_id(self, sample_ids):
        with torch.no_grad():
            return self.loss_values[sample_ids].cpu().numpy()
    

def train_model_coteaching(model, co_model, optimizer, co_optimizer, scheduler, co_scheduler, num_epochs, dataloaders, rate_schedule,
                           dataset_sizes, device, scaler=None, effective_phase=['train', 'val']):
    since = time.time()
    best_acc = 0
    # scaler = torch.cuda.amp.GradScaler()
    filtered_idx_catach_main = torch.tensor([], dtype=torch.int64).to(device)
    filtered_idx_catach_co = torch.tensor([], dtype=torch.int64).to(device)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            running_loss_co = 0.0
            running_acc1_co = 0.0
            running_acc5_co = 0.0

            # Iterate over data.
            for inputs, labels, idx in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    idx = idx.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    co_optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        co_outputs = co_model(inputs)
                        loss, co_loss, idx_filtered_main, idx_filtered_co = loss_coteaching(outputs, co_outputs, labels, rate_schedule[epoch-1], idx)
                        filtered_idx_catach_main = torch.cat((filtered_idx_catach_main, idx_filtered_main))
                        filtered_idx_catach_co = torch.cat((filtered_idx_catach_co, idx_filtered_co))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                
                                scaler.scale(loss).backward()
                                scaler.scale(co_loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.step(co_optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                co_loss.backward()
                                optimizer.step()
                                co_optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_co += co_loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()
                acc1_co, acc5_co = accuracy(co_outputs, labels, topk=(1, 5))
                running_acc1_co += acc1_co.item()
                running_acc5_co += acc5_co.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            epoch_loss_co = running_loss_co / dataset_sizes[phase]
            epoch_acc_co = running_acc1_co / dataset_sizes[phase]
            epoch_acc5_co = running_acc5_co / dataset_sizes[phase]
            best_acc = max(epoch_acc, epoch_acc_co, best_acc)
            if phase == 'train':
                scheduler.step()
                co_scheduler.step()

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
            print('{} Co Loss: {:.4f} Co Acc top 1: {:.4f} Co Acc top 5: {:.4f}'.format(
                phase, epoch_loss_co, epoch_acc_co, epoch_acc5_co))
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    # torch.save(model.state_dict(), model_save_path)
    idx_save_path = './analyses/' + model_save_path + '_catched_idx.pth'
    save_catached_tensor(filtered_idx_catach_main, idx_save_path)
    co_idx_save_path = './analyses/'+ model_save_path + '_co_catched_idx.pth'
    save_catached_tensor(filtered_idx_catach_co, co_idx_save_path)
    return model, co_model, model_save_path

def train_model_coteaching_plus(model, co_model, optimizer, co_optimizer, scheduler, co_scheduler, num_epochs, dataloaders, rate_schedule,
                           dataset_sizes, device, scaler=None, effective_phase=['train', 'val']):
    since = time.time()
    best_acc = 0
    # scaler = torch.cuda.amp.GradScaler()
    filtered_idx_catach_main = torch.tensor([], dtype=torch.int64).to(device)
    filtered_idx_catach_co = torch.tensor([], dtype=torch.int64).to(device)
    
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        step = 0
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            running_loss_co = 0.0
            running_acc1_co = 0.0
            running_acc5_co = 0.0
            # Iterate over data.
            for inputs, labels, idx in tqdm(dataloaders[phase]):
                step += len(inputs)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    idx = idx.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    co_optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        co_outputs = co_model(inputs)
                        loss, co_loss, idx_filtered_main, idx_filtered_co = loss_coteaching_plus(outputs, co_outputs, labels, rate_schedule[epoch-1], idx, step=0)
                        filtered_idx_catach_main = torch.cat((filtered_idx_catach_main, idx_filtered_main))
                        filtered_idx_catach_co = torch.cat((filtered_idx_catach_co, idx_filtered_co))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                
                                scaler.scale(loss).backward()
                                scaler.scale(co_loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.step(co_optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                co_loss.backward()
                                optimizer.step()
                                co_optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_co += co_loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()
                acc1_co, acc5_co = accuracy(co_outputs, labels, topk=(1, 5))
                running_acc1_co += acc1_co.item()
                running_acc5_co += acc5_co.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            epoch_loss_co = running_loss_co / dataset_sizes[phase]
            epoch_acc_co = running_acc1_co / dataset_sizes[phase]
            epoch_acc5_co = running_acc5_co / dataset_sizes[phase]
            best_acc = max(epoch_acc, epoch_acc_co, best_acc)
            if phase == 'train':
                scheduler.step()
                co_scheduler.step()

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
            print('{} Co Loss: {:.4f} Co Acc top 1: {:.4f} Co Acc top 5: {:.4f}'.format(
                phase, epoch_loss_co, epoch_acc_co, epoch_acc5_co))
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    # torch.save(model.state_dict(), model_save_path)
    idx_save_path = './analyses/' + model_save_path + '_catched_idx.pth'
    save_catached_tensor(filtered_idx_catach_main, idx_save_path)
    co_idx_save_path = './analyses/'+ model_save_path + '_co_catched_idx.pth'
    save_catached_tensor(filtered_idx_catach_co, co_idx_save_path)
    return model, co_model, model_save_path

def train_model_mentor(model, optimizer, scheduler, num_epochs, dataloaders, rate_schedule,
                           dataset_sizes, device, scaler=None, effective_phase=['train', 'val']):
    since = time.time()
    best_acc = 0
    # scaler = torch.cuda.amp.GradScaler()
    filtered_idx_catach_main = torch.tensor([], dtype=torch.int64).to(device)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            # Iterate over data.
            for inputs, labels, idx in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    idx = idx.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss, idx_filtered_main = loss_mentor(outputs, labels, rate_schedule[epoch-1], idx)
                        filtered_idx_catach_main = torch.cat((filtered_idx_catach_main, idx_filtered_main))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                
                                scaler.scale(loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
            else:
                if (epoch+1)//10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1)//40 == 0:
                        torch.save(model.state_dict(), model_save_path)
                best_acc = max(epoch_acc, best_acc)
            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
            
            gc.collect()
            torch.cuda.empty_cache()
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    # torch.save(model.state_dict(), model_save_path)
    idx_save_path = model_save_path + '_mentor_idx.pth'
    save_catached_tensor(filtered_idx_catach_main, idx_save_path)
    return model, model_save_path

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, scaler=None,
                effective_phase=['train', 'val']):
    """
    Duplicate with the mentor model when setting forget rate to 0, just for comparsion if the loss is correct
    """
    since = time.time()
    best_acc = 0
    
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            # Iterate over data.
            for inputs, labels, _ in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                # loss.backward()
                                scaler.scale(loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                _, preds = torch.max(outputs, 1)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
            else:
                if (epoch+1)//10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1)//40 == 0:
                        torch.save(model.state_dict(), model_save_path)
            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path

def train_model_transition(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, transition_matrix, scaler=None,
                effective_phase=['train', 'val']):
    """
    Has bug in the loss function, should be fixed.
    Aiming to train the model with transition matrix
    """
    since = time.time()
    best_acc = 0
    # scaler = torch.cuda.amp.GradScaler()
    #noise_traisition_matrix = transition_matrix.unsqueeze(0).repeat(dataloaders['train'].batch_size, 1, 1)
    loss_fn = loss_transition_matrix(transition_matrix)#torch.nn.CrossEntropyLoss() #loss_transition_matrix(transition_matrix)# #
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    warm_up = 5
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            # Iterate over data.
            for inputs, labels, _ in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # just for testing
                        # out_prob = torch.nn.functional.softmax(outputs, dim=1)
                        # adjusted_outputs = torch.matmul(out_prob, transition_matrix)
                        ce_loss = ce_loss_fn(outputs, labels)
                        loss = loss_fn(outputs, labels)
                        total_loss = max((warm_up-epoch)/warm_up, 0)*ce_loss + min(1,epoch/warm_up)*loss
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                # loss.backward()
                                scaler.scale(total_loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                _, preds = torch.max(outputs, 1)
                # statistics
                running_loss += total_loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
            else:
                if (epoch+1)//10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1)//40 == 0:
                        torch.save(model.state_dict(), model_save_path)
            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path

def train_model_SAM(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, scaler=None,
                effective_phase=['train', 'val'], rho=0.04):
    """
    Training model using Sharpness-Aware Minimization (SAM) (Foret et al, 2022) loss
    """

    since = time.time()
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        for phase in effective_phase:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            for inputs, labels, _ in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    if phase == 'train':
                        # First forward-backward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        # Compute ε(w)
                        with torch.no_grad():
                            grad_w = [p.grad.data for p in model.parameters() if p.grad is not None]
                            grad_norm = torch.norm(
                                torch.stack([torch.norm(g.detach(), dtype=torch.float32) for g in grad_w]),
                                dtype=torch.float32
                            )
                            scale = (rho / (grad_norm + 1e-5)).to(dtype=torch.float16)
                        
                        # Perturb the model
                        with torch.no_grad():
                            for p, g in zip(model.parameters(), grad_w):
                                p.add_(g, alpha=scale)
                        
                        # Second forward-backward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        
                        # Revert the perturbation
                        with torch.no_grad():
                            for p, g in zip(model.parameters(), grad_w):
                                p.sub_(g, alpha=scale)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            if phase == 'val':
                if (epoch+1) % 10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1) % 40 == 0:
                        torch.save(model.state_dict(), model_save_path)

    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path

def train_model_AUM(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device,scaler=None,
                effective_phase=['train', 'val']):
    """
    Duplicate with the mentor model when setting forget rate to 0, just for comparsion if the loss is correct
    """
    since = time.time()
    best_acc = 0
    
    aum_calculator = EfficientAUMCalculator(num_samples=dataset_sizes['train'], device=device)
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            # Iterate over data.
            for inputs, labels, indexes in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                # loss.backward()
                                scaler.scale(loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()
                aum_calculator.update(outputs, labels, indexes)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
            else:
                if (epoch+1)//10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1)//40 == 0:
                        torch.save(model.state_dict(), model_save_path)
            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path, aum_calculator

def get_predictions(model, dataloader, device):
    model.eval()
    predictions = torch.tensor(()).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for inputs, labels, idx in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions = torch.cat((predictions, preds))
    return predictions

def calculate_pc(prev_predictions, current_predictions):
    comparsion_1 = prev_predictions == current_predictions
    pc = 1-torch.sum(comparsion_1).item() / len(prev_predictions)
    return pc

def train_model_LW(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, add_loader, scaler=None,
                effective_phase=['train', 'val']):
    since = time.time()
    best_acc = 0
    best_model = None
    
    # Label Wave parameters
    k = 3  # Moving average window
    pc_history = deque(maxlen=k)
    previous_predictions = None
    best_pc = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        for phase in effective_phase:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            for inputs, labels, _ in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            if scaler is not None:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            if phase == 'train':
                scheduler.step()
        
        # Label Wave: Get predictions on the whole dataset
        current_predictions = get_predictions(model, add_loader, device)
        
        if previous_predictions is not None:
            pc = calculate_pc(previous_predictions, current_predictions)
            pc_history.append(pc)
            print(f"Prediction Changes: {pc}")
            
            if len(pc_history) == k:
                pc_avg = np.mean(pc_history)
                print(f"Moving Average PC: {pc_avg}")
                
                if pc_avg < best_pc:
                    best_pc = pc_avg
                    best_model = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("Early stopping")
                    model.load_state_dict(best_model)
                    return model, str(int(since)) + '_' + '{:4f}'.format(best_acc)
        
        previous_predictions = current_predictions
        
        # Save model periodically
        if epoch % 10 == 0 or epoch % 40 == 0:
            model_save_path = str(int(since)) + '_' + f'epoch_{epoch}'
            torch.save(model.state_dict(), model_save_path)

    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path

def init_meta_net(dataloaders, device):
    """
    Initialize the meta-net for meta-learning
    """
    meta_model = base_meta_network(dataloaders['train'].dataset.__classes__()).to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    meta_scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=7, gamma=0.1)
    return meta_model, meta_optimizer, meta_scheduler

def train_model_margin(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, scaler=None,
                effective_phase=['train', 'val'], meta=True):
    """
    Training the meta-learning model with margin loss
    """
    since = time.time()
    best_acc = 0
    warmup_epochs = 5
    # init meta-net
    meta_model, meta_optimizer, meta_scheduler = init_meta_net(dataloaders, device)
    batch_size = dataloaders['train'].batch_size
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in effective_phase:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            # meta learning
            if meta and phase == 'train' and epoch > warmup_epochs:
                meta_model.train()
                print('Meta-learning')
                for inputs, labels, _ in dataloaders['meta']:
                    # First step
                    with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
                        model.train()
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                            outputs = fmodel(inputs)
                            margin = meta_model()
                            margin = margin.repeat(inputs.size(0), 1)
                            first_loss = criterion(outputs + margin, labels)
                        if scaler is not None:
                            scaler.scale(first_loss).backward()
                            scaler.step(diffopt)
                            scaler.update()
                        else:
                            #print('First backward')
                            first_loss.backward(retain_graph=True)
                            diffopt.step(first_loss)
                        #del inputs, labels, outputs, first_loss
                        # clear cuda cache
                        #torch.cuda.empty_cache()
                    
                    # Second step
                    #for inputs, labels, _ in dataloaders['meta']:
                        #print('Second step')
                            # inputs = inputs.to(device)
                            # labels = labels.to(device)
                        meta_optimizer.zero_grad()
                        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                            model.eval()
                            #print('Second forward')
                            outputs2 = fmodel(inputs)
                            second_loss = criterion(outputs2, labels)
                    
                    # Update meta model
                        
                        if scaler is not None:
                            scaler.scale(second_loss).backward()
                            scaler.step(meta_optimizer)
                            scaler.update()
                        else:
                            #print('Second backward') # Just for tracking the process
                            second_loss.backward()
                            meta_optimizer.step()
                            meta_scheduler.step()
                        #del inputs, labels, outputs, second_loss
                        # clear cuda cache
                        #torch.cuda.empty_cache()

            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0

            # Iterate over data.
            
            for inputs, labels, _ in tqdm(dataloaders[phase]):
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        if meta and phase == 'train' and epoch > warmup_epochs:
                            
                            margin = meta_model()
                            margin = margin.repeat(inputs.size(0), 1)
                            
                            loss = criterion(outputs + margin, labels)
                        else:
                            loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if scaler is not None:
                                # loss.backward()
                                scaler.scale(loss).backward()
                                # optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                _, preds = torch.max(outputs, 1)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # should be changed to offical accuracy code
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                running_acc1 += acc1.item()
                running_acc5 += acc5.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc1 / dataset_sizes[phase]
            epoch_acc5 = running_acc5 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
            else:
                if (epoch+1)//10 == 0:
                    model_save_path = str(int(since)) + '_' + f'epoch_{epoch+1}'
                    if epoch <= 40:
                        torch.save(model.state_dict(), model_save_path)
                    elif (epoch+1)//40 == 0:
                        torch.save(model.state_dict(), model_save_path)
            print('{} Loss: {:.4f} Acc top 1: {:.4f} Acc top 5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc5))
            # clear cuda cache
    
    model_save_path = str(int(since)) + '_' + '{:4f}'.format(best_acc)
    return model, model_save_path
    