import os
import sys
import time
import gc
import copy
import random
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from image_util import *
from data_util import * 
from env import *
from sum_tree import *
import model_creator as mCreator

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = sum_tree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    def __len__(self):
        return int(self.tree.n_entries)
    
class DQN_Agent:   
    def __init__(self, global_features_net, local_features_net, num_epoch, num_actions, num_fc_nodes, lr=0.0001, lr_decay=0.01, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99, guided_eps=0.5, mem_size=10000, batch_size=32, iter_update_target = 300, use_gpu = True, use_chex_net = False, transforms = None):
        #Initiate agent components
        self.num_actions = num_actions
        self.evaluate_net = mCreator.Bi_DQN(num_actions, global_features_net, local_features_net, num_fc_nodes, True, use_chex_net)
        self.target_net = mCreator.Bi_DQN(num_actions, global_features_net, local_features_net, num_fc_nodes, True, use_chex_net)
        self.evaluate_net.train(True)
        self.target_net.train(True)
        if(use_gpu):
            #self.evaluate_net = torch.nn.DataParallel(self.evaluate_net.cuda(), device_ids=[0, 1])
            #self.target_net = torch.nn.DataParallel(self.target_net.cuda(), device_ids=[0, 1])
            self.evaluate_net = self.evaluate_net.cuda()
            self.target_net = self.target_net.cuda()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=lr, weight_decay=lr_decay)
        self.loss_function = nn.MSELoss()
        self.memory = ReplayMemory(capacity=mem_size)
        self.transforms = transforms
        
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon_linear_factor = float(self.epsilon-self.min_epsilon)/float(num_epoch)
        self.guided_eps = guided_eps
        self.gamma = gamma
        self.iter_update_target = iter_update_target
 
        self.iter_counter = 0
    #@profile
    def select_action(self, current_env, bb_img):
        sample = np.random.uniform()
        if(sample > self.epsilon):
            #Exploitation
            #print("Exploitation")
            #Convert to 3 channel and apply torchvision transforms
            #unsqueeze for batch size
            global_img = transform_img_for_model(current_env.full_env)
            if(self.transforms is not None):
                global_img = self.transforms(global_img)
            global_img.unsqueeze_(0)
            global_img = global_img.to(0)
            bb_img = transform_img_for_model(bb_img)
            if(self.transforms is not None):
                bb_img = self.transforms(bb_img)
            bb_img.unsqueeze_(0)
            bb_img = bb_img.to(0)
            q = self.evaluate_net(global_img, bb_img)
            action = torch.max(q, 1)[1].data[0]
            return action
        else:
            #Exploration
            sample = np.random.uniform()
            if(sample > self.guided_eps):
                #print("Random exploration")
                #Random exploration
                return random.randint(0, self.num_actions-1)
            else:
                #print("Guided exploration")
                #Guided exploration
                rewards = []
                for i in range(self.num_actions):
                    rewards.append(current_env.step_foresee(i))
                pos_reward_index = []
                zero_reward_index = []
                for i in range(len(rewards)):
                    if(rewards[i] > 0):
                        pos_reward_index.append(i)
                    if(rewards[i] == 0):
                        zero_reward_index.append(i)
                if(len(pos_reward_index) > 0):
                    return random.choice(pos_reward_index)
                elif(len(zero_reward_index) > 0):
                    return random.choice(zero_reward_index)
                else:
                    return random.randint(0, self.num_actions-1)
    #For inference once agent is trained
    def select_action_infer(self, current_env, bb_img):
        global_img = transform_img_for_model(current_env.full_env)
        bb_img = transform_img_for_model(bb_img)
        if(self.transforms is not None):
            global_img = self.transforms(global_img)
            bb_img = self.transforms(bb_img)
        q = self.target_net(global_img, bb_img)
        action = torch.max(q, 1)[1].data[0]
        return action

    def store_transitions(self, state_tup, action, reward, next_state_tup, done):
        self.memory.push((state_tup, action, reward, next_state_tup, done))
    #@profile
    def learn(self):
        self.iter_counter += 1
        if(len(self.memory) < self.batch_size):
            return
        #Random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)
        batch_state_env = []
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_next_env = []
        batch_state_next_state = []
        batch_done = []
        for t in transitions:
            (bse, bs), ba, br, (bsne, bsns), bd = t
            bse = transform_img_for_model(bse)
            bs = transform_img_for_model(bs)
            bsne = transform_img_for_model(bsne)
            bsns = transform_img_for_model(bsns)
            if(self.transforms is not None):
                bse = self.transforms(bse)
                bs = self.transforms(bs)
                bsne = self.transforms(bsne)
                bsns = self.transforms(bsns)
            batch_state_env.append(bse)
            batch_state.append(bs)
            batch_action.append(ba)
            batch_reward.append(br)
            batch_state_next_env.append(bsne)
            batch_state_next_state.append(bsns)
            batch_done.append(bd)
            
        batch_state = Variable(torch.stack(batch_state)).cuda(async=True)
        batch_state_env = Variable(torch.stack(batch_state_env)).cuda(async=True)
        batch_action = torch.FloatTensor(batch_action).unsqueeze_(0)
        batch_action = batch_action.view(batch_action.size(1), -1)
        batch_action = Variable(batch_action).cuda(async=True)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze_(0)
        batch_reward = batch_reward.view(batch_reward.size(1), -1)
        batch_reward = Variable(batch_reward).cuda(async=True)
        batch_next_state = Variable(torch.stack(batch_state_next_state)).cuda(async=True)
        batch_state_next_env = Variable(torch.stack(batch_state_next_env)).cuda(async=True)

        # current Q values are estimated by NN for all actions
        current_q_values = self.evaluate_net(batch_state_env, batch_state).gather(1, batch_action.long())
      
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.target_net(batch_state_next_env, batch_next_state).detach().max(1)[0]
        max_next_q_values = max_next_q_values.unsqueeze_(0)
        max_next_q_values = max_next_q_values.view(max_next_q_values.size(1), -1)
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)
     
        # loss is measured from error between current and newly expected Q values
        loss = self.loss_function(current_q_values, expected_q_values)

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #free variables
        del batch_state, batch_state_env, batch_action, batch_reward, batch_next_state, batch_state_next_env, loss
        
        if(self.iter_counter % self.iter_update_target == 0):
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
    #Per Episode
    def decay_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = new_epsilon
    #Per Epoch
    def decay_epsilon_linear(self):
        self.epsilon -= self.epsilon_linear_factor
        if(self.epsilon < self.min_epsilon):
            self.epsilon = self.min_epsilon 
    
    def set_training(self, train_or_not=True):
        if(train_or_not):
            self.evaluate_net.train(True)
            self.target_net.train(True)
        else:
            self.evaluate_net.train(False)
            self.target_net.train(False)
            self.evaluate_net.eval()
            self.target_net.eval()
            
class DQN_Agent_Single_Net:   
    def __init__(self, features_net, num_epoch, num_actions, num_fc_nodes, lr=0.0001, lr_decay=0.01, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99, guided_eps=0.5, mem_size=10000, batch_size=32, iter_update_target = 300, use_gpu = True, use_chex_net = False, transforms = None, use_pr_replay=True):
        #Initiate agent components
        self.num_actions = num_actions
        self.evaluate_net = mCreator.DQN(num_actions, features_net, num_fc_nodes, True, use_chex_net)
        self.target_net = mCreator.DQN(num_actions, features_net, num_fc_nodes, True, use_chex_net)
        self.evaluate_net.train(True)
        self.target_net.train(True)
        if(use_gpu):
            #self.evaluate_net = torch.nn.DataParallel(self.evaluate_net.cuda(), device_ids=[0, 1])
            #self.target_net = torch.nn.DataParallel(self.target_net.cuda(), device_ids=[0, 1])
            self.evaluate_net = self.evaluate_net.cuda()
            self.target_net = self.target_net.cuda()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=lr, weight_decay=lr_decay)
        #self.loss_function = nn.MSELoss()
        self.use_pr_replay = use_pr_replay
        if(use_pr_replay):
            self.memory = PrioritizedReplayMemory(mem_size)
        else:
            self.memory = ReplayMemory(capacity=mem_size)
        self.loss_function = nn.SmoothL1Loss().cuda()
        self.transforms = transforms
        
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon_linear_factor = float(self.epsilon-self.min_epsilon)/float(num_epoch)
        self.guided_eps = guided_eps
        self.gamma = gamma
        self.iter_update_target = iter_update_target
        self.use_ddqn = True
        self.iter_counter = 0
    #@profile
    def select_action(self, current_env, bb_img):
        sample = np.random.uniform()
        if(sample > self.epsilon):
            #Exploitation
            #print("Exploitation")
            #Convert to 3 channel and apply torchvision transforms
            #unsqueeze for batch size
            bb_img = transform_img_for_model(bb_img.numpy(), self.transforms)
            #bb_img = transform_img_for_model(bb_img)
            #if(self.transforms is not None):
            #    bb_img = self.transforms(bb_img)
            bb_img.unsqueeze_(0)
            bb_img = bb_img.to(0)
            q = self.evaluate_net(bb_img)
            action = torch.max(q, 1)[1].data[0]
           
            return action
        else:
            #Exploration
            sample = np.random.uniform()
            if(sample > self.guided_eps):
                #print("Random exploration")
                #Random exploration
                return random.randint(0, self.num_actions-1)
            else:
                #print("Guided exploration")
                #Guided exploration
                rewards = []
                for i in range(self.num_actions):
                    rewards.append(current_env.step_foresee(i))
                pos_reward_index = []
                zero_reward_index = []
                for i in range(len(rewards)):
                    if(rewards[i] > 0):
                        pos_reward_index.append(i)
                    if(rewards[i] == 0):
                        zero_reward_index.append(i)
                if(len(pos_reward_index) > 0):
                    return random.choice(pos_reward_index)
                elif(len(zero_reward_index) > 0):
                    return random.choice(zero_reward_index)
                else:
                    return random.randint(0, self.num_actions-1)
    #For inference once agent is trained
    def select_action_infer(self, current_env, bb_img):
        bb_img = transform_img_for_model(bb_img.numpy(), self.transforms)
        #bb_img = transform_img_for_model(bb_img)
        #if(self.transforms is not None):
        #    bb_img = self.transforms(bb_img)
        bb_img.unsqueeze_(0)
        bb_img = bb_img.to(0)
        q = self.target_net(bb_img)
        action = torch.max(q, 1)[1].data[0]
        return action

    def store_transitions(self, state_tup, action, reward, next_state_tup, done):
        if(self.use_pr_replay):
            target = self.evaluate_net(transform_img_for_model(state_tup.numpy(), self.transforms).unsqueeze_(0).to(0)).data
            old_val = target[0][action]
            target_val = self.target_net(transform_img_for_model(next_state_tup.numpy(), self.transforms).unsqueeze_(0).to(0)).data
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(target_val)

            error = abs(old_val - target[0][action])
            self.memory.add(error, (state_tup, action, reward, next_state_tup, done))
        else:
            self.memory.push((state_tup, action, reward, next_state_tup, done))
    
    @profile
    def learn(self):
        self.iter_counter += 1
        if(len(self.memory) < self.batch_size):
            return
        #Random transition batch is taken from experience replay memory
        if(self.use_pr_replay):
            transitions, idxs, is_weights = self.memory.sample(self.batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)
        batch_state = []
        batch_action = []        
        batch_reward = []
        batch_state_next_state = []
        batch_done = []
        for t in transitions:
            bs, ba, br, bsns, bd = t
            bs = transform_img_for_model(bs.numpy(), self.transforms)
            #bs = transform_img_for_model(bs)
            #if(self.transforms is not None):
            #    bs = self.transforms(bs)
            batch_state.append(bs)
            batch_action.append(ba)
            batch_reward.append(br)
            bsns = transform_img_for_model(bsns.numpy(), self.transforms)
            #bsns = transform_img_for_model(bsns)
            #if(self.transforms is not None):
            #    bsns = self.transforms(bsns)
            batch_state_next_state.append(bsns)
            batch_done.append(bd)
        
        with torch.no_grad():    
            batch_state = Variable(torch.stack(batch_state).cuda(async=True))
            batch_next_state = Variable(torch.stack(batch_state_next_state).cuda(async=True))
            batch_action = torch.FloatTensor(batch_action).unsqueeze_(0)
            batch_action = batch_action.view(batch_action.size(1), -1)
            batch_action = Variable(batch_action.cuda(async=True))
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze_(0) 
            batch_reward = batch_reward.view(batch_reward.size(1), -1)
            batch_reward = Variable(batch_reward.cuda(async=True))
            batch_dones = torch.FloatTensor(batch_done).unsqueeze_(0)
            batch_dones = batch_dones.view(batch_dones.size(1), -1)
            batch_dones = Variable(batch_dones.cuda(async=True))
 
        # current Q values are estimated by NN for all actions
        current_q_values = self.evaluate_net(batch_state).gather(1, batch_action.long())
        # expected Q values are estimated from actions which gives maximum Q value
        if(self.use_ddqn):
            next_actions = torch.max(self.evaluate_net(batch_next_state),1)[1].detach()
            next_actions = next_actions.unsqueeze_(0)
            next_actions = next_actions.view(next_actions.size(1), -1)
            max_next_q_values = self.target_net(batch_next_state).gather(1, next_actions.long()).detach()
        else:
            max_next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
            max_next_q_values = max_next_q_values.unsqueeze_(0)
            max_next_q_values = max_next_q_values.view(max_next_q_values.size(1), -1)
        expected_q_values = batch_reward + (1 - batch_dones) * (self.gamma * max_next_q_values)
        # loss is measured from error between current and newly expected Q values
        #torch.cuda.synchronize()
        
        loss = self.loss_function(current_q_values, expected_q_values)
        
        errors = torch.abs(expected_q_values - current_q_values).cpu().data.numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        #print(loss)
        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #free variables
        del batch_state, batch_action, batch_reward, batch_next_state, transitions, current_q_values, max_next_q_values, expected_q_values
        #for obj in gc.get_objects():
        #    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #        print(type(obj), obj.size())        
        #if(self.iter_counter % self.iter_update_target == 0):
        #    self.target_net.load_state_dict(self.evaluate_net.state_dict())
        return loss

    #Per Episode
    def decay_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = new_epsilon
    #Per Epoch
    def decay_epsilon_linear(self):
        self.epsilon -= self.epsilon_linear_factor
        if(self.epsilon < self.min_epsilon):
            self.epsilon = self.min_epsilon 
    
    def set_training(self, train_or_not=True):
        if(train_or_not):
            self.evaluate_net.train(True)
            self.target_net.train(True)
        else:
            self.evaluate_net.train(False)
            self.target_net.train(False)
            self.evaluate_net.eval()
            self.target_net.eval()


