import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from image_util import *
from data_util import * 
from env import *
from dqn_agent_old import *
import model_creator_old as mCreator


#G10 Server
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT_DATA_PATH = "/home/g10/medical/data/rsna_pneumonia/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
"""

"""
ROOT_DATA_PATH = "/home/ubuntu/richielo/data/pneumo_data/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
S1_TEST_IMG_PATH = ROOT_DATA_PATH + "stage_1_test_images"
"""

#AZURE
ROOT_DATA_PATH = "/home/ai_hku_iris/Desktop/richielo/data/"
S1_TRAIN_IMG_PATH = ROOT_DATA_PATH + "stage_1_train_images"
S1_TRAIN_ANNO_PATH = ROOT_DATA_PATH + "stage_1_train_labels.csv"
S1_TEST_IMG_PATH = ROOT_DATA_PATH + "stage_1_test_images"


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
MEM_SIZE = 10000
ITER_UPDATE_TARGET = 300

#Global Variables
NUM_ACTIONS = 9
ACTION_THRES = 0.2
TRIGGER_THRES = 0.6
TRIGGER_REWARD = 10
INIT_BB_THRES = 0.8
steps_done = 0
INFER_TIMESTEP = 25

def run_inferences(parsed_dict, num_epoch, infer_step, target_net, global_features_net, local_features_net, target_model_path, use_chexnet=False, use_gpu=True):
    #train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  
    #train_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = None
    
    #Initialize dqn agent
    dqn_agent = DQN_Agent(global_features_net, local_features_net, num_epoch, NUM_ACTIONS, HIDDEN_LAYER, LR, LR_DECAY, GAMMA, EPS_START, EPS_MIN, EPS_DECAY, GUIDED_EPS, MEM_SIZE, BATCH_SIZE, ITER_UPDATE_TARGET, use_gpu, use_chexnet, train_transforms)
    #Load trained dqn
    checkpoint = torch.load(target_model_path)
    dqn_agent.target_net.load_state_dict(checkpoint)
    
    dqn_agent.set_training(False)
    
    #Iterate testing data
    for k, v in parsed_dict.items():
        pid = k
        env = PneumoEnv(parsed_dict[pid], False, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
        done = False
        running_steps = 0
        found_boxes = []
        while not done:
            action = dqn_agent.select_action_infer(env, env.extract_bound_box_image(env.bb))
            state_bb, action, next_state_bb, done = env.step_infer(action)
            running_steps += 1
            if done:
                #Found box - reset box, running_step and done to continue?
                dqn_agent.global_features = None
                found_boxes.append(env.bb)
            elif(running_steps == infer_step):
                #No box found
                dqn_agent.global_features = None
                break
    pass

def run_inferences_single(parsed_dict, num_epoch, infer_step, features_net, target_model_path, use_chexnet=False, use_gpu=True, use_end_loc=True):
    #train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    #train_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = None
    
    #Initialize dqn agent
    dqn_agent = DQN_Agent_Single_Net(features_net, num_epoch, NUM_ACTIONS, HIDDEN_LAYER, LR, LR_DECAY, GAMMA, EPS_START, EPS_MIN, EPS_DECAY, GUIDED_EPS, MEM_SIZE, BATCH_SIZE, ITER_UPDATE_TARGET, use_gpu, use_chexnet, train_transforms)
    
    #Load trained dqn
    checkpoint = torch.load(target_model_path)
    dqn_agent.target_net.load_state_dict(checkpoint)
    
    dqn_agent.set_training(False)
    
    #Iterate testing data
    result_dict_list = []
    with torch.no_grad():
        for k, v in parsed_dict.items():
            result_dict = {}
            pid = k
            result_dict['patientId'] = pid
            print(parsed_dict[pid])
            env = PneumoEnv(parsed_dict[pid], False, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
            done = False
            running_steps = 0
            found_boxes = []
            while not done:
                action = dqn_agent.select_action_infer(env, env.extract_bound_box_image(env.bb))
                state_bb, action, next_state_bb, done = env.step_infer(action)
                running_steps += 1
                if done:
                    #Found box - reset box, running_step and done to continue?
                    found_boxes.append(env.bb)
                elif(running_steps == infer_step):
                    #No box found
                    if(use_end_loc):
                        found_boxes.append(env.bb)
                    break
            if(len(found_boxes) > 0):
                result_dict["PredictionString"] = "1.0" + " " + str(found_boxes[0][0]) + " " + str(found_boxes[0][1]) + " " + str(found_boxes[0][2]) + " " + str(found_boxes[0][3])
            else:
                result_dict["PredictionString"] = ""
            result_dict_list.append(result_dict)
        result_df = pd.DataFrame(result_dict_list)
        result_df.to_csv("dqn_train_singular_final2_test_submission.csv", sep=',', encoding='utf-8')
        print("Number of test images:" + str(len(parsed_dict)))
        print("Number of found boxes: " + str(len(found_boxes)))
        print("Number of test images:" + str(len(parsed_dict)))
        print("Number of found boxes: " + str(len(found_boxes)))

if __name__ == '__main__':
    cwd = os.getcwd()
    use_chexnet = False
    data_dict = parse_test_dataset(S1_TEST_IMG_PATH)
    
    #Get pretrained global and local features net
    if(use_chexnet):
        #global_features_net = mCreator.get_chexnet(14, True, "chexnet_pretrained.pth.tar")
        #local_features_net = mCreator.get_chexnet(14, True, "chexnet_pretrained.pth.tar")
        print("Loaded pretrained chestxnet")
    else:
        #global_features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_global2_b24_f0_e20_001_9_imgnet_normalize.pt", True)
        #local_features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_local1_b4_f0_e60_00001_9_imgnet_normalize.pt", False)
        features_net = mCreator.get_resnet50_model(2, True, 0, "pretrain_local1_b4_f0_e60_00001_9_imgnet_normalize.pt", False)
        print("Loaded pretrained resnet")
    
    run_inferences_single(data_dict, 20, INFER_TIMESTEP, features_net, "checkpoint_2_noscaleinc.pt", use_chexnet=False, use_gpu=True)
