import os 
import sys
import numpy as np
from PIL import Image
import torch

#TODO - add save function, these functions can be used to check movement 
def crop_image(image_array, bb):
    image_array_copy = image_array.clone()
    y_min = int(bb[0])
    x_min = int(bb[1])
    height = int(bb[2])
    width = int(bb[3])
    y_max = y_min + height
    x_max = x_min + width
    return image_array[y_min:y_max, x_min:x_max]

#Keep image size, set pixel value outside of bounding box as 0
def crop_pad_image(image_array, bb):
    image_array_copy = image_array.clone()
    y_min = int(bb[0])
    x_min = int(bb[1])
    height = int(bb[2])
    width = int(bb[3])
    y_max = y_min + height
    x_max = x_min + width
    mask_array = np.zeros(image_array.shape, dtype=int)
    mask_array[y_min:y_max, x_min:x_max] = 1
    zero_array = np.where(mask_array==0)
    image_array_copy[zero_array[0],zero_array[1]] = 0
    return image_array_copy
    
def set_bb_to_black(image_array, bb):
    image_array_copy = image_array.clone()
    y_min = int(bb[0])
    x_min = int(bb[1])
    height = int(bb[2])
    width = int(bb[3])
    y_max = y_min + height
    x_max = x_min + width
    mask_array = np.zeros(image_array.shape, dtype=int)
    mask_array[y_min:y_max, x_min:x_max] = 1
    zero_array = np.where(mask_array==1)
    image_array_copy[zero_array[0],zero_array[1]] = 0
    return image_array_copy

def transform_img_for_model(image_array, transforms=None):
    image_array_copy = np.copy(image_array)
    #image_array_copy.unsqueeze_(0)
    image_array_copy = np.expand_dims(image_array_copy, axis=2)
    if(transforms is None):
        image_array_copy = torch.from_numpy(image_array_copy).repeat(3, 1, 1)
    else:
        image_array_copy = transforms(image_array_copy).repeat(3, 1, 1)
    return image_array_copy

def save_image_from_tensor(image_array, path):
    og = Image.fromarray(image_array.numpy())
    og = og.convert('RGB')
    og.save(path)
    
def resize_image(image_array, width, height):
    og = Image.fromarray(image_array.numpy())
    og = og.convert('RGB')
    og = og.resize((width, height))
    og = og.convert('L')
    return np.array(og)
