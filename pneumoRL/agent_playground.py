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
import torchvision.transforms as transforms

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
                save_image_from_tensor(env.extract_bound_box_image(env.bb), "move_test/test_taller_fatter_start.jpeg")
                for i in range(2):
                    state_bb, action, reward, next_state_bb, done = env.step(7)
                for i in range(2):
                    state_bb, action, reward, next_state_bb, done = env.step(6)
                save_image_from_tensor(env.extract_bound_box_image(env.bb), "move_test/test_taller_fatter_end.jpeg")
                exit()
    return dqn_agent
    
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
    #epoch-batch size-trigger reward thres-lr-decay-hidden layer size
    model_name = "dqn_train_singular3_e{}_b{}_t0.6_1e6_99_{}_basicnorm_epslinear_resnet_pretrained.pt".format(num_epoch, BATCH_SIZE, HIDDEN_LAYER)
    torch.save(trained_agent.target_net.state_dict(), cwd+'/' + model_name)
    
    """                                                  
    #model = mCreator.get_resnet50_model(2, True, 0)
    #model2 = mCreator.get_resnet50_model(2, True, 0)
    #dqn = mCreator.Bi_DQN(9, model, model2, 512)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    #new_env = PneumoEnv(testDataDict, True, 0.1, 0.6, 10, 0.7)
    #print(np.argmax(dqn(new_env.full_env, new_env.extract_bound_box_image()).detach().numpy()))
    #print(torch.argmax(dqn(new_env.full_env, new_env.extract_bound_box_image())))
    trained_agent = run_episodes(num_epoch, train_loader, test_loader, global_features_net, local_features_net)
    model_name = "dqn_train1_b{}_0001_99_{}_chexnet_pretrained.pt".format(BATCH_SIZE, HIDDEN_LAYER)
    torch.save(trained_agent.target_net.state_dict(), cwd+'/' + model_name)
    
    #Load train image paths and annotations
    anns = pd.read_csv(open(os.path.join(ROOT_DATA_PATH, 'stage_1_train_labels.csv'), 'r'))
    parsed_training_data = parse_dataset(S1_TRAIN_IMG_PATH, anns)
    print("Number of train images: " + str(len(parsed_training_data)))
    for k, v in parsed_training_data.items():
        print(k)
        print(v)
        testImg = get_dicom_image_data(parsed_training_data[k]['dicom'])
        print(np.amax(testImg))
        print(np.amin(testImg))
        print(testImg.shape)
        break
    """
