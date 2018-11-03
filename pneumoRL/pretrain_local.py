import os
import sys
import time
import copy
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler

from data_util import *
import model_creator as mCreator

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

ROOT_DATA_PATH = "/home/ubuntu/richielo/pneumoRL/local_patches/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"

def train(model, train_set, val_set, train_loader, val_loader, num_epochs, optimizer, scheduler, criterion, use_gpu=True):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    if(use_gpu):
        #model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1])
        model = model.cuda(0)
        
    #Use cross validation
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("Training batch {}/{}".format(i, train_batches), end='\n', flush=True)
            
            """
            # Use half training dataset
            if i >= train_batches / 2:
                break
            """
            
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
      
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data).item()

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train  / len(train_set)
        avg_acc = acc_train  / len(train_set)
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(val_loader):
            if i % 100 == 0:
                print("Validation batch {}/{}".format(i, val_batches), end='\n', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(0), volatile=True), Variable(labels.cuda(0), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data).item()
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(val_set)
        avg_acc_val = acc_val / len(val_set)
        
        epoch_time = time.time() - epoch_start
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Time used: {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

#TODO
"""
1. Balanced sampling
2. Normalization
3. Augmentation
"""
if __name__ == '__main__':
    cwd = os.getcwd()
    split_ratio = 0.1
    num_classes = 2
    num_freeze = 0
    num_epoch = 20
    padded = True
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]), transforms.RandomRotation([0,360])])
    #no - transforms.Normalize([0.5], [0.5])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    #Load train image patches paths and annotations
    train_dict, val_dict = create_balanced_dataset_local(ROOT_DATA_PATH, padded, split_ratio)
    pneumo_train_dataset = PneumoLocalClassificationDataset(train_dict, transform)
    pneumo_val_dataset = PneumoLocalClassificationDataset(val_dict, transform)
    img = get_jpeg_image_data(pneumo_train_dataset.image_paths[0])
    #No need split training and validation, should use whole dataset to train?
    #Weighted sampler for training
    train_labels = np.array(pneumo_train_dataset.labels)
    class_sample_count = np.array([len(np.where(train_labels==t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    train_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    train_loader = DataLoader(pneumo_train_dataset, batch_size=6, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(pneumo_val_dataset, batch_size=4, num_workers=2, pin_memory=True)
    
    model = mCreator.get_resnet50_model(num_classes, True, num_freeze)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train(model, pneumo_train_dataset, pneumo_val_dataset, train_loader, val_loader, num_epoch, optimizer_ft, exp_lr_scheduler, criterion)
    torch.save(model.state_dict(), cwd+'/pretrain_local1_b8_f0_e60_00001_9_imgnet_normalize.pt')