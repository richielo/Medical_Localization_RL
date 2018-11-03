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
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False

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
LR_DECAY = 0.0
HIDDEN_LAYER = 512  # NN hidden layer size
BATCH_SIZE = 48  # Q-learning batch size
MEM_SIZE = 8000
ITER_UPDATE_TARGET = 500

#Global Variables
NUM_ACTIONS = 9
ACTION_THRES = 0.1
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
        losses = []
        learn_step = 1
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("Training batch {}/{}".format(i, train_batches), end='\n', flush=True)
                dqn_agent.set_training(False)
                with torch.no_grad():
                    test_return_list = []

                    for i, data in enumerate(test_loader):
                        if i % 100 == 0:
                            print("Testing batch {}/{}".format(i, test_batches), end='\n', flush=True)
                        patient_ids = data
                        for j in range(len(patient_ids)):
                            env = PneumoEnv(parsed_dict[patient_ids[j]], True, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
                            R = 0
                            done = False
                            running_steps = 0
                            actions = []
                            while not done:
                                action = dqn_agent.select_action_infer(env, env.extract_bound_box_image(env.bb))
                                state_bb, action, reward, next_state_bb, done = env.step(action)
                                running_steps += 1
                                R += reward
                                actions.append(action)
                                if running_steps >= INFER_TIMESTEP:
                                    no_result_penalty = -10
                                    R += no_result_penalty
                                    done = True
                                if done:
                                    test_return_list.append(R)
                                    print("Test Episode:", i, "R:", R)
                                    print("actions: " + str(actions))
                                    sys.stdout.flush()
                dqn_agent.set_training(True)

            patient_ids = data
            for j in range(len(patient_ids)):
                env = PneumoEnv(parsed_dict[patient_ids[j]], True, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
                R = 0
                done = False
                running_step = 0
                while not done:
                    #curr_bb = env.extract_bound_box_image(env.bb)
                    action = dqn_agent.select_action(env, env.extract_bound_box_image(env.bb))
                    state_bb, action, reward, next_state_bb, done = env.step(action)
                    num_train_step += 1
                    running_step += 1
                    dqn_agent.store_transitions(env.extract_bound_box_image(state_bb), action, reward, env.extract_bound_box_image(next_state_bb), done) 
                    R += reward
                    if(num_train_step % learn_freq == 0):
                        #learn_start = time.time()
                        loss = dqn_agent.learn()
                        print("Loss: " + str(loss))
                        if loss is not None:
                            losses.append(loss.item())
                        #learn_end = time.time()
                        #print("learn time for one pass-through: " + str(learn_end - learn_start))
                        learn_step += 1
                    if(running_step >= 25):
                        R += -10
                        done = True
                    if done:
                        dqn_agent.target_net.load_state_dict(dqn_agent.evaluate_net.state_dict())
                        train_return_list.append(R)
                        print("Epoch:", epoch, "Epsilon:", dqn_agent.epsilon, "R:", R)
                        print("Length of memory" + str(len(dqn_agent.memory)))
                        print("mem usage: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                        sys.stdout.flush()
                    #del curr_bb
                    gc.collect()
                    torch.cuda.empty_cache()
                    
        epoch_end = time.time()
        print("Epoch time: " + str(epoch_end-epoch_start))
        print("Average loss: " + str(sum(losses)/ float(len(losses))))
                
        #Done one epoch
        dqn_agent.decay_epsilon_linear()
        #Validation episodes
        if((epoch+1) % 1 == 0):
            dqn_agent.set_training(False)
            with torch.no_grad():
                test_return_list = []
                for i, data in enumerate(test_loader):
                    if i % 100 == 0:
                        print("Testing batch {}/{}".format(i, test_batches), end='\n', flush=True)
                    patient_ids = data
                    for j in range(len(patient_ids)):
                        env = PneumoEnv(parsed_dict[patient_ids[j]], True, ACTION_THRES, TRIGGER_THRES, TRIGGER_REWARD, INIT_BB_THRES)
                        R = 0
                        done = False
                        running_steps = 0
                        actions = []
                        while not done:
                            action = dqn_agent.select_action_infer(env, env.extract_bound_box_image(env.bb))
                            state_bb, action, reward, next_state_bb, done = env.step(action)
                            running_steps += 1
                            R += reward
                            actions.append(action)
                            if running_steps >= INFER_TIMESTEP:
                                no_result_penalty = -10
                                R += no_result_penalty
                                done = True 
                            if done:
                                test_return_list.append(R)
                                print("Test Episode:", i, "R:", R)
                                print("actions: " + str(actions))
                                sys.stdout.flush()
                                
                avg_train_reward = sum(train_return_list)/float(len(train_return_list))
                avg_test_reward = sum(test_return_list)/float(len(test_return_list))
                print("Average training reward: " + str(avg_train_reward))
                print("Average testing reward: " + str(avg_test_reward))
                if(best_reward is None):
                    best_model_wts = copy.deepcopy(dqn_agent.target_net.state_dict())
                    best_reward = avg_test_reward
                elif(avg_test_reward > best_reward):
                    best_model_wts = copy.deepcopy(dqn_agent.target_net.state_dict())
                    best_reward = avg_test_reward
                    
    cwd = os.getcwd()
    torch.save(dqn_agent.target_net.state_dict(), cwd+'/' + "dqn_singular_train_3_end_net.pt")
    dqn_agent.target_net.load_state_dict(best_model_wts)
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
    model_name = "dqn_train_singular3_e{}_b{}_t0.6_1e1_99_{}_basicnorm_epslinear_resnet_pretrained.pt".format(num_epoch, BATCH_SIZE, HIDDEN_LAYER)
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
