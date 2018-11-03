"""
Author: Richie Lo 

Description: this script generates patches to pretrain local patches of various sizes (Definitely need augmentation when training)

class                         Target
Lung Opacity                  1          8964
No Lung Opacity / Not Normal  0         11500
Normal                        0          8525

Each positive image, each box - produce 2 positive patches
Each negative image - produce 1
min area of gt boxes: 2320, 48*48
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from data_util import *
from image_util import *
from env import * 
from PIL import Image

"""
#G10 Server
ROOT_DATA_PATH = "/home/g10/medical/data/rsna_pneumonia/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
"""

ROOT_DATA_PATH = "/home/ubuntu/richielo/data/pneumo_data/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"

"""
    #Env movement testing 
    test_env = PneumoEnv(v, True, 0.1, 0.7)
    img = get_dicom_image_data(v['dicom'])
    save_image_from_tensor(img, 'move_test/og.jpeg')
    print(test_env.bb)
    bb_img_1 = crop_pad_image(img, test_env.bb)
    save_image_from_tensor(bb_img_1, 'move_test/bb1.jpeg')
    test_env.step(7)
    print(test_env.bb)
    bb_img_2 = crop_pad_image(img, test_env.bb)
    save_image_from_tensor(bb_img_2, 'move_test/bb2.jpeg')
    
    #Image Utils testing
    img = get_dicom_image_data(v['dicom'])
    boxes = v['boxes']
    cropped = crop_image(img, boxes[0])
    cropped_pad = crop_pad_image(img, boxes[0])
    og = Image.fromarray(img.numpy())
    c = Image.fromarray(cropped.numpy())
    cp = Image.fromarray(cropped_pad.numpy())
    og = og.convert('RGB')
    c = c.convert('RGB')
    cp = cp.convert('RGB')
    og.save("crop_test/test_og.jpeg")
    c.save("crop_test/c.jpeg")
    cp.save("crop_test/cp.jpeg")
"""

def generate_pos_patch(v, pos_num):    
    gt_boxes = v['boxes']
    pos_patches = []
    for b in gt_boxes:
        num_b_found = 0
        while(num_b_found < pos_num):
            rand_y = random.randint(0, 1024)
            rand_x = random.randint(0, 1024)
            height = random.randint(0, 1024-rand_y)
            width = random.randint(0, 1024-rand_x)
            cand_box = [rand_y, rand_x, height, width]
            if(calculate_iou(b, cand_box) >= 0.7):
                #found
                num_b_found += 1
                pos_patches.append(cand_box)
    assert len(gt_boxes)*pos_num == len(pos_patches)
    return pos_patches

def generate_neg_patch(v, neg_num):
    min_area = 48 * 48
    neg_patches = []
    num_b_found = 0
    while(num_b_found < neg_num):
        rand_y = random.randint(0, 1024)
        rand_x = random.randint(0, 1024)
        height = random.randint(0, 1024-rand_y)
        width = random.randint(0, 1024-rand_x)
        cand_box = [rand_y, rand_x, height, width]
        if(width*height >= min_area):
            #found
            num_b_found += 1
            neg_patches.append(cand_box)
    return neg_patches

def generate_patches(dataset, pos_num, neg_num):
    i = 0
    for k, v in dataset.items():
        if(v['label'] == 1):
            pos_patches = generate_pos_patch(v, pos_num)
            #save patches (both padded and cropped version)
            for m in range(len(pos_patches)):
                img = get_dicom_image_data(v['dicom'])
                cropped = crop_image(img, pos_patches[m])
                cropped_pad = crop_pad_image(img, pos_patches[m])
                #patientid_label_patchindex
                save_image_from_tensor(cropped, 'local_patches/cropped/'+str(k)+"_1_"+str(m)+".jpeg")
                save_image_from_tensor(cropped_pad, 'local_patches/cropped_pad/'+str(k)+"_1_"+str(m)+".jpeg")
        else:
            neg_patches = generate_neg_patch(v, neg_num)
            #save patches (both padded and cropped version)
            for m in range(len(neg_patches)):
                img = get_dicom_image_data(v['dicom'])
                cropped = crop_image(img, neg_patches[m])
                cropped_pad = crop_pad_image(img, neg_patches[m])
                #patientid_label_patchindex
                save_image_from_tensor(cropped, 'local_patches/cropped/'+str(k)+"_0_"+str(m)+".jpeg")
                save_image_from_tensor(cropped_pad, 'local_patches/cropped_pad/'+str(k)+"_0_"+str(m)+".jpeg")
        i += 1
        if(i%1000 == 0):
            print("generated patches for {} images".format(i))

if __name__ == '__main__':
    cwd = os.getcwd()
    
    #Load train image paths and annotations
    anns = pd.read_csv(open(os.path.join(ROOT_DATA_PATH, 'stage_1_train_labels.csv'), 'r'))
    parsed_training_data = parse_dataset(S1_TRAIN_IMG_PATH, anns)
    
    pos_num = 2
    neg_num = 1
    generate_patches(parsed_training_data, pos_num, neg_num)