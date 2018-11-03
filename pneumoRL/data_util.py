import os
import sys

import random
import math
import numpy as np
import cv2
import json
import pydicom
import pandas as pd 
import glob
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsample import StratifiedSampler

def get_all_paths_by_filetype(path, filetype):
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            #".jpeg" for image
            if file.endswith(filetype):
                 paths.append(os.path.join(root, file))
    return paths

def get_dicom_data(path):
    """
    Given dicom data file path, 
    return the corresponding dicom data.
    """
    return pydicom.read_file(path)

def get_dicom_image_data(path):
    """
    Given dicom data file path, 
    return the corresponding image data.
    """
    dicomFile = get_dicom_data(path)
    return torch.from_numpy(dicomFile.pixel_array).float()

def get_jpeg_image_data(path):
    img = Image.open(path)
    return torch.from_numpy(np.array(img)).float()

def parse_dataset(data_path, df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]
    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': data_path + '/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

def parse_test_dataset(data_path):
    data_dict = {}
    paths = get_all_paths_by_filetype(data_path, ".dcm")
    for p in paths:
        pid = p.split('/')[-1].replace('.dcm', '')
        data_dict[pid] = {"dicom":p}
    assert len(data_dict) == len(paths)
    return data_dict

"""
Dataset classes for classification pretraining
"""
class PneumoClassificationDataset(Dataset):
    def __init__(self, parsedData, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for k, v in parsedData.items():
            self.image_paths.append(v['dicom'])
            self.labels.append(v['label'])
    def __getitem__(self, index):
        img = get_dicom_image_data(self.image_paths[index])
        img.unsqueeze_(0)
        img = img.repeat(3, 1, 1)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.image_paths)

class PneumoLocalClassificationDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform 
        for k, v in data_dict.items():
            self.image_paths.append(k)
            self.labels.append(v)
        assert len(self.image_paths) == len(self.labels)
    def __getitem__(self, index):
        img = get_jpeg_image_data(self.image_paths[index])
        img = img.reshape(img.size(2), img.size(1), -1)
        label = self.labels[index]
        return img, label 
    def __len__(self):
        return len(self.image_paths)
"""
Dataset classe for localization training
"""
class PneumoLocalizationDataset(Dataset):
    def __init__(self, parsedData, transform=None):
        self.patient_ids = []
        self.transform = transform
        for k, v in parsedData.items():
            if(v['label'] == 1):
                self.patient_ids.append(k)
    def __getitem__(self, index):
        return (self.patient_ids[index])
    def __len__(self):
        return len(self.patient_ids)    
"""
class PneumoLocalizationDataset(Dataset):
    def __init__(self, parsedData, transform=None):
        self.image_paths = []
        self.labels = []
        self.boxes = []
        self.transform = transform
        for k, v in parsedData.items():
            if(v['label'] == 1):
                self.image_paths.append(v['dicom'])
                self.boxes.append(v['boxes'])
                self.labels.append(v['label'])
    def __getitem__(self, index):
        return self.image_paths[index], self.boxes[index], self.labels[index]
    def __len__(self):
        return len(self.image_paths)  
"""
    
def create_balanced_dataset(parsed_dict, split_ratio):
    total_data_num = len(parsed_dict.items())
    val_size = int(total_data_num*split_ratio)
    train_dict = {}
    val_dict = {}
    pos_keys = []
    neg_keys = []
    for k, v in parsed_dict.items():
        if(v['label'] == 1):
            pos_keys.append(k)
        else:
            neg_keys.append(k)
    print("Numer of positive samples: " + str(len(pos_keys)))
    print("Number of negative samples: " + str(len(neg_keys)))
    val_pos_size = int(len(pos_keys)/total_data_num*val_size)
    val_neg_size = val_size - val_pos_size
    random.shuffle(pos_keys)
    random.shuffle(neg_keys)
    while(len(val_dict) < val_pos_size):
        key = pos_keys.pop()
        val_dict[key] = parsed_dict[key]
    while(len(val_dict) < val_size):
        key = neg_keys.pop()
        val_dict[key] = parsed_dict[key]
    for pos_key in pos_keys:
        train_dict[pos_key] = parsed_dict[pos_key]
    for neg_key in neg_keys:
        train_dict[neg_key] = parsed_dict[neg_key]
    print("train dict size: " + str(len(train_dict.items())))
    print("val dict size: " + str(len(val_dict.items())))
    
    return train_dict, val_dict

def create_balanced_dataset_local(data_path, padded, split_ratio):
    image_paths = []
    labels = []
    data_dict = {}
    pos_keys = []
    neg_keys = []
    if(padded):
        data_path = data_path + "cropped_pad"
    else:
        data_path = data_path + "cropped"
    paths = get_all_paths_by_filetype(data_path, ".jpeg")
    total_data_num = len(paths)
    val_size = int(total_data_num*split_ratio)
    train_dict = {}
    val_dict = {}
    for p in paths:
        image_paths.append(p)
        filename = p.split('/')[-1].replace(".jpeg", "")
        label = int(filename.split('_')[1])
        labels.append(label)
        data_dict[p] = label
        if(label == 1):
            pos_keys.append(p)
        else:
            neg_keys.append(p)
    assert len(image_paths) == len(labels)
    val_pos_size = int(len(pos_keys)/total_data_num*val_size)
    val_neg_size = val_size - val_pos_size
    random.shuffle(pos_keys)
    random.shuffle(neg_keys)
    while(len(val_dict) < val_pos_size):
        key = pos_keys.pop()
        val_dict[key] = data_dict[key]
    while(len(val_dict) < val_size):
        key = neg_keys.pop()
        val_dict[key] = data_dict[key]
    for pos_key in pos_keys:
        train_dict[pos_key] = data_dict[pos_key]
    for neg_key in neg_keys:
        train_dict[neg_key] = data_dict[neg_key]
    
    return train_dict, val_dict
