import numpy as np
import os
import sys
import gc
import time
import copy
import resource
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from image_util import *
from data_util import * 
from env import *
from dqn_agent import *
import model_creator as mCreator

"""
#G10 Server
os.environ["CUDA_VISIBLE_DEVICES"]="1"
ROOT_DATA_PATH = "/home/g10/medical/data/rsna_pneumonia/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
"""

"""
#AWS
ROOT_DATA_PATH = "/home/ubuntu/richielo/data/pneumo_data/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
"""

#AZURE
ROOT_DATA_PATH = "/home/ai_hku_iris/Desktop/richielo/data/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"

"""
Attempt to visualize feature maps (specifically for resnet50)
"""
def return_sequential(layer_num, model):
    return nn.Sequential(
            *list(model.children())[:layer_num]
        )

class get_activation_layer(nn.Module):
    def __init__(self, model, total_layers):
        super().__init__()
        self.model = model
        self.total_layers = total_layers
        self.layer_models = []
        for i in range(self.total_layers):
             self.layer_models.append(return_sequential(i, self.model))
    def forward(self, x):
        self.outputs = []
        for i in range(self.total_layers):
            self.outputs.append(self.layer_models[i](x))
        return self.outputs
    
def visulaize_layers(outputs):
    for index, layer in enumerate(outputs):
        features = layer.data
        size_plot = features.shape[1]
        if size_plot % 2 != 0:
            size_plot += 1
        original_size = np.int(np.ceil(np.sqrt(size_plot)))
        f, axarr = plt.subplots(original_size + 1, original_size + 1)
        i, j = 0,0
        counter = 1
        for blocks in features:
            for block in blocks:
                counter += 1
                x = block.cpu().numpy()
                if counter % original_size == 0:
                    i += 1
                    j = 0 

                axarr[i,j].imshow(x)
                j += 1
        counter = 0
        print(f'layer {index} done')
        f.savefig(f'layer_op/output{index}.jpg')
        print('image generated')

#Hyperparameters
EPISODES = 200  # number of episodes
EPS_START = 1.0  # e-greedy threshold start value
GUIDED_EPS = 0.5
EPS_MIN = 0.1  # e-greedy threshold end value
EPS_DECAY = 0.99  # e-greedy threshold decay
GAMMA = 0.99  # Q-learning discount factor
LR = 1e-6  # NN optimizer learning rate
LR_DECAY = 0.01
HIDDEN_LAYER = 1024  # NN hidden layer size
BATCH_SIZE = 48  # Q-learning batch size
MEM_SIZE = 8000
ITER_UPDATE_TARGET = 30

#Global Variables
NUM_ACTIONS = 9
ACTION_THRES = 0.2
TRIGGER_THRES = 0.6
TRIGGER_REWARD = 10
INIT_BB_THRES = 0.75
steps_done = 0
INFER_TIMESTEP = 20

@profile
def run_episodes(num_epoch, parsed_dict, train_loader, test_loader, features_net, use_chexnet=False, use_gpu=True):
    since = time.time()
    best_reward = None
    
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    #train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    train_transforms = transforms.Compose([transforms.ToTensor()])
    #normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  
    #train_transforms = None
    #Initialize dqn agent
    dqn_agent = DQN_Agent_Single_Net(features_net, num_epoch, NUM_ACTIONS, HIDDEN_LAYER, LR, LR_DECAY, GAMMA, EPS_START, EPS_MIN, EPS_DECAY, GUIDED_EPS, MEM_SIZE, BATCH_SIZE, ITER_UPDATE_TARGET, use_gpu, use_chexnet, train_transforms)
    best_model_wts = copy.deepcopy(dqn_agent.target_net.state_dict())
    feature_net = dqn_agent.target_net.feature_net
    
    learn_freq = 10

    #Training
    for epoch in range(num_epoch):
        dqn_agent.set_training(True)
        num_train_step = 0
        train_return_list = []
        epoch_start = time.time()
        print("Epoch {}/{}".format(epoch, num_epoch))
        print('-' * 10)
        learn_step = 1
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("Training batch {}/{}".format(i, train_batches), end='\n', flush=True)
            patient_ids = data
            for j in range(len(patient_ids)):
                env = PneumoEnv(parsed_dict[patient_ids[j]], True, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
                nr_layers = 10
                tmp_model = get_activation_layer(feature_net, nr_layers)
                img = transform_img_for_model(env.extract_bound_box_image(env.bb), train_transforms)
                layer_outputs = tmp_model(Variable(img.unsqueeze_(0).to(0)))
                visulaize_layers(layer_outputs)
                exit()
    
if __name__ == '__main__':
    cwd = os.getcwd()
    split_ratio = 0.05
    num_epoch = 10
    use_chexnet = False
    
    #Load train image paths and annotations
    anns = pd.read_csv(open(os.path.join(ROOT_DATA_PATH, 'stage_1_train_labels.csv'), 'r'))
    parsed_training_data_dict = parse_dataset(S1_TRAIN_IMG_PATH, anns)
    full_dataset = PneumoLocalizationDataset(parsed_training_data_dict)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=0, pin_memory=False)

    #Get pretrained global and local features net
    if(use_chexnet):
        #global_features_net = mCreator.get_chexnet(14, True, "chexnet_pretrained.pth.tar")
        #local_features_net = mCreator.get_chexnet(14, True, "chexnet_pretrained.pth.tar")
        features_net = mCreator.get_chexnet(14, True, "chexnet_pretrained.pth.tar")
        print("Loaded pretrained chestxnet")
    else:
        #global_features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_global2_b24_f0_e20_001_9_imgnet_normalize.pt", True)
        #local_features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_local1_b4_f0_e60_00001_9_imgnet_normalize.pt", False)
        features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_local4_no_resize_resnet_b6_f0_e30_0001_9_imgnet_normalize.pt", False)
        #features_net = mCreator.get_vgg19_model(2, True, 0, "pretrain_local2_vgg19_b8_f0_e60_00001_9_imgnet_normalize.pt", False)
        print("Loaded pretrained resnet")
    #Training
    trained_agent = run_episodes(num_epoch, parsed_training_data_dict, train_loader, test_loader, features_net, use_chexnet, True)