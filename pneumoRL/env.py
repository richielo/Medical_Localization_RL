import os
import sys
import random

from data_util import *
from image_util import *

"""
2 methods for local data
1. crop and pad
2. crop and resize
"""
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
#Adjustable 
MIN_WIDTH = 10
MIN_HEIGHT = 10



def init_bounding_box(img_shape, coverage):
    factor = math.floor(math.sqrt(img_shape[0] * img_shape[1] * coverage))
    pos_bb = {1:[0,0, factor, factor], 2:[IMG_HEIGHT-factor-1, 0, factor, factor], 3:[IMG_HEIGHT-factor-1, IMG_WIDTH-factor-1, factor, factor], 4:[0, IMG_WIDTH-factor-1, factor, factor]}
    return pos_bb[random.randint(1,4)]

def calculate_iou(bb1, bb2):
    bb1x1 = bb1[1]
    bb1x2 = bb1[1] + bb1[3]
    bb1y1 = bb1[0]
    bb1y2 = bb1[0] + bb1[2]
    bb2x1 = bb2[1]
    bb2x2 = bb2[1] + bb2[3]
    bb2y1 = bb2[0]
    bb2y2 = bb2[0] + bb2[2]
    
    x1 = max(bb1x1, bb2x1)
    y1 = max(bb1y1, bb2y1)
    x2 = min(bb1x2, bb2x2)
    y2 = min(bb1y2, bb2y2)
    
    inter_buf_x = x2-x1+1
    inter_buf_y = y2-y1+1
    if(inter_buf_x <= 0 or inter_buf_y <= 0):
        return 0 
    else:
        inter_area = max(0, inter_buf_x) * max(0, inter_buf_y)
        box1Area = (bb1x2 - bb1x1 + 1) * (bb1y2 - bb1y1 + 1)
        box2Area = (bb2x2 - bb2x1 + 1) * (bb2y2 - bb2y1 + 1)
        iou = inter_area / float(box1Area + box2Area - inter_area)
        return iou

def calculate_manhattan_distance(bb1, bb2):
    bb1x1 = bb1[1]
    bb1x2 = bb1[1] + bb1[3]
    bb1y1 = bb1[0]
    bb1y2 = bb1[0] + bb1[2]
    bb1x_center = int((bb1x1+bb1x2)/2.0)
    bb1y_center = int((bb1y1+bb1y2)/2.0)
    bb2x1 = bb2[1]
    bb2x2 = bb2[1] + bb2[3]
    bb2y1 = bb2[0]
    bb2y2 = bb2[0] + bb2[2]
    bb2x_center = int((bb2x1+bb2x2)/2.0)
    bb2y_center = int((bb2y1+bb2y2)/2.0)
    
    return abs(bb2y_center-bb1y_center) + abs(bb2x_center-bb1x_center)
    
"""
This defines the environment and interactions between it and the agent. Agent has control over the bounding box defined within the environment
"""
class PneumoEnv():
    
    def __init__(self, dataDict, training, action_thres, trigger_thres, trigger_reward, init_bb_thres):
        #TODO - action thres and starting coverage - tunable parameter
        #Load the image's pixel array as the environment
        self.full_env = get_dicom_image_data(dataDict['dicom'])
        self.gt_boxes = None
        self.target = None
        self.label = None
        self.action_threshold = action_thres
        self.trigger_threshold = trigger_thres
        self.trigger_reward = trigger_reward
        self.init_bb_threshold = init_bb_thres
        self.is_finished = False
        #Initialize bounding box - randomly on 4 corners of the image, covering 80% of the image
        #(y, x, height, width)
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
        #History vector
        if(training):
            #Load ground truth bounding box(es)
            self.gt_boxes = dataDict['boxes']
            self.label = dataDict['label']
            if(self.label == 1):
                self.target_bb = self.gt_boxes[random.randint(0,len(self.gt_boxes)-1)]
    
    def get_current_state(self):
        """
        returns current bounding box's padded image + full_env
        """
        pass
    
    def get_reward(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        if(action == 8):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                return 1.0
            else:
                return -1.0

    def get_reward_mod(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        oldbb_man_dist = calculate_manhattan_distance(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        newbb_man_dist = calculate_manhattan_distance(newBb, self.target_bb)
        
        if(action == 8):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            reward = 0.0 
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                reward += 1.0
            else:
                reward += -1.0
            man_dist_diff = newbb_man_dist - oldbb_man_dist
            if(man_dist_diff < 0):
                reward += 1.0
            else:
                reward += -1.0
            return reward

    #@profile
    def step_foresee(self, action):
        #Forsee results of an action for guided exploration, without updating the environment
        old_bb = self.bb.copy()
        new_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            new_bb[1] -= a_x
            if(new_bb[1] < 0):
                new_bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            new_bb[1] += a_x
            if(new_bb[1] + new_bb[3] > IMG_WIDTH - 1):
                new_bb[1] = IMG_WIDTH - 1 - new_bb[3]
        elif(action == 2):
            #Vertical - Up
            new_bb[0] += a_y
            if(new_bb[0] + new_bb[2] > IMG_HEIGHT - 1):
                new_bb[0] = IMG_HEIGHT - 1 - new_bb[2]
        elif(action == 3):
            #Vertical - Down
            new_bb[0] -= a_y
            if(new_bb[0] < 0):
                new_bb[0] = 0
        elif(action == 4):
            #Scale - increase
            new_bb[1] -= a_x
            new_bb[3] += 2 * a_x
            if(new_bb[1] < 0):
                new_bb[1] = 0
            if(new_bb[1] + new_bb[3] > IMG_WIDTH - 1):
                new_bb[3] = IMG_WIDTH - 1 - new_bb[1]
            new_bb[0] -= a_y
            new_bb[2] += 2 * a_y
            if(new_bb[0] < 0):
                new_bb[0] = 0
            if(new_bb[0] + new_bb[2] > IMG_HEIGHT - 1):
                new_bb[2] = IMG_HEIGHT - 1 - new_bb[0]
        elif(action == 5):
            #Scale - decrease
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - fatter
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 7):
            #Aspect ratio - taller
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
                
        reward = self.get_reward_mod(action, old_bb, new_bb)
        return reward
    #@profile
    def step(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - increase
            self.bb[1] -= a_x
            self.bb[3] += 2 * a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[3] = IMG_WIDTH - 1 - self.bb[1]
            self.bb[0] -= a_y
            self.bb[2] += 2 * a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[2] = IMG_HEIGHT - 1 - self.bb[0]
        elif(action == 5):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 7):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 8):
            #Trigger
            self.is_finished = True
        
        reward = self.get_reward_mod(action, old_bb, self.bb)
        return old_bb, action, reward, self.bb, self.is_finished
    
    def step_infer(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - increase
            self.bb[1] -= a_x
            self.bb[3] += 2 * a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[3] = IMG_WIDTH - 1 - self.bb[1]
            self.bb[0] -= a_y
            self.bb[2] += 2 * a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[2] = IMG_HEIGHT - 1 - self.bb[0]
        elif(action == 5):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 7):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 8):
            #Trigger
            self.is_finished = True
        
        return old_bb, action, self.bb, self.is_finished

    def reset_bb(self):
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
    
    def black_out(self):
        self.full_env = set_bb_to_black(self.full_env, self.bb)
    
    def extract_bound_box_image(self, bb):
        """
        extracts pixel content of current bounding box
        """
        bb_img = crop_pad_image(self.full_env, bb)
        return bb_img
    
class PneumoEnv2():
    """
    Without scale increase
    """
    
    def __init__(self, dataDict, training, action_thres, trigger_thres, trigger_reward, init_bb_thres):
        #TODO - action thres and starting coverage - tunable parameter
        #Load the image's pixel array as the environment
        self.full_env = get_dicom_image_data(dataDict['dicom'])
        self.gt_boxes = None
        self.target = None
        self.label = None
        self.action_threshold = action_thres
        self.trigger_threshold = trigger_thres
        self.trigger_reward = trigger_reward
        self.init_bb_threshold = init_bb_thres
        self.is_finished = False
        #Initialize bounding box - randomly on 4 corners of the image, covering 80% of the image
        #(y, x, height, width)
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
        #History vector
        if(training):
            #Load ground truth bounding box(es)
            self.gt_boxes = dataDict['boxes']
            self.label = dataDict['label']
            if(self.label == 1):
                self.target_bb = self.gt_boxes[random.randint(0,len(self.gt_boxes)-1)]
    
    def get_current_state(self):
        """
        returns current bounding box's padded image + full_env
        """
        pass
    
    def get_reward(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        if(action == 7):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                return 1.0
            elif iou_diff == 0:
                return 0.0
            else:
                return -1.0
    
    def get_reward_mod(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        oldbb_man_dist = calculate_manhattan_distance(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        newbb_man_dist = calculate_manhattan_distance(newBb, self.target_bb)
        
        if(action == 7):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            reward = 0.0 
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                reward += 1.0
            elif iou_diff == 0:
                reward += 0.0
            else:
                reward += -1.0
            man_dist_diff = newbb_man_dist - oldbb_man_dist
            if(man_dist_diff < 0):
                reward += 1.0
            elif man_dist_diff == 0:
                reward += 0.0
            else:
                reward += -1.0 
            return reward
    #@profile
    def step_foresee(self, action):
        #Forsee results of an action for guided exploration, without updating the environment
        old_bb = self.bb.copy()
        new_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            new_bb[1] -= a_x
            if(new_bb[1] < 0):
                new_bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            new_bb[1] += a_x
            if(new_bb[1] + new_bb[3] > IMG_WIDTH - 1):
                new_bb[1] = IMG_WIDTH - 1 - new_bb[3]
        elif(action == 2):
            #Vertical - Up
            new_bb[0] += a_y
            if(new_bb[0] + new_bb[2] > IMG_HEIGHT - 1):
                new_bb[0] = IMG_HEIGHT - 1 - new_bb[2]
        elif(action == 3):
            #Vertical - Down
            new_bb[0] -= a_y
            if(new_bb[0] < 0):
                new_bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
                
        reward = self.get_reward_mod(action, old_bb, new_bb)
        return reward
    #@profile
    def step(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 7):
            #Trigger
            self.is_finished = True
        
        reward = self.get_reward_mod(action, old_bb, self.bb)
        return old_bb, action, reward, self.bb, self.is_finished
    
    def step_infer(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 7):
            #Trigger
            self.is_finished = True
        
        return old_bb, action, self.bb, self.is_finished

    def reset_bb(self):
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
    
    def black_out(self):
        self.full_env = set_bb_to_black(self.full_env, self.bb)
    
    def extract_bound_box_image(self, bb):
        """
        extracts pixel content of current bounding box
        """
        bb_img = crop_pad_image(self.full_env, bb)
        return bb_img

    
class PneumoEnv3():
    """
    Without scale increase with corner-pivotal scaling
    """
    
    def __init__(self, dataDict, training, action_thres, trigger_thres, trigger_reward, init_bb_thres):
        #TODO - action thres and starting coverage - tunable parameter
        #Load the image's pixel array as the environment
        self.full_env = get_dicom_image_data(dataDict['dicom'])
        self.gt_boxes = None
        self.target = None
        self.label = None
        self.action_threshold = action_thres
        self.trigger_threshold = trigger_thres
        self.trigger_reward = trigger_reward
        self.init_bb_threshold = init_bb_thres
        self.is_finished = False
        #Initialize bounding box - randomly on 4 corners of the image, covering 80% of the image
        #(y, x, height, width)
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
        #History vector
        if(training):
            #Load ground truth bounding box(es)
            self.gt_boxes = dataDict['boxes']
            self.label = dataDict['label']
            if(self.label == 1):
                self.target_bb = self.gt_boxes[random.randint(0,len(self.gt_boxes)-1)]
    
    def get_current_state(self):
        """
        returns current bounding box's padded image + full_env
        """
        pass
    
    def get_reward(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        if(action == 7):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                return 1.0
            elif iou_diff == 0:
                return 0.0
            else:
                return -1.0
    
    def get_reward_mod(self, action, oldBb, newBb):
        """
        * Assume must have ground truth boxes for training, can consider using terminate action, may destabalize training
        2 separate reward schemes, depends on whether there are ground truth boxes or not
        1. If has ground truth boxes, calculates reward based of IOU
        2. Reward based on the number of steps it takes until it determines there is no candidate box
        """
        oldbb_iou = calculate_iou(oldBb, self.target_bb)
        oldbb_man_dist = calculate_manhattan_distance(oldBb, self.target_bb)
        newbb_iou = calculate_iou(newBb, self.target_bb)
        newbb_man_dist = calculate_manhattan_distance(newBb, self.target_bb)
        
        if(action == 7):
            if(newbb_iou >= self.trigger_threshold):
                return self.trigger_reward
            else:
                return -1 * self.trigger_reward
        else:
            reward = 0.0 
            iou_diff = newbb_iou - oldbb_iou
            if iou_diff > 0:
                reward += 1.0
            else:
                reward += -1.0
            man_dist_diff = newbb_man_dist - oldbb_man_dist
            if(man_dist_diff < 0):
                reward += 1.0
            else:
                reward += -1.0
            return reward
    #@profile
    def step_foresee(self, action):
        #Forsee results of an action for guided exploration, without updating the environment
        old_bb = self.bb.copy()
        new_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            new_bb[1] -= a_x
            if(new_bb[1] < 0):
                new_bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            new_bb[1] += a_x
            if(new_bb[1] + new_bb[3] > IMG_WIDTH - 1):
                new_bb[1] = IMG_WIDTH - 1 - new_bb[3]
        elif(action == 2):
            #Vertical - Up
            new_bb[0] += a_y
            if(new_bb[0] + new_bb[2] > IMG_HEIGHT - 1):
                new_bb[0] = IMG_HEIGHT - 1 - new_bb[2]
        elif(action == 3):
            #Vertical - Down
            new_bb[0] -= a_y
            if(new_bb[0] < 0):
                new_bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            new_bb[0] += a_y
            new_bb[2] -= 2 * a_y
            if(new_bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - new_bb[2]) / 2
                new_bb[0] -= new_fac
                new_bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            new_bb[1] += a_x
            new_bb[3] -= 2 * a_x
            if(new_bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - new_bb[3]) / 2
                new_bb[1] -= new_fac
                new_bb[3] += 2 * new_fac
                
        reward = self.get_reward_mod(action, old_bb, new_bb)
        return reward
    #@profile
    def step(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 7):
            #Trigger
            self.is_finished = True
        
        reward = self.get_reward_mod(action, old_bb, self.bb)
        return old_bb, action, reward, self.bb, self.is_finished
    
    def step_infer(self, action):
        """
        executes action selected by the agent
        """
        old_bb = self.bb.copy()
        a_x = int(self.action_threshold * self.bb[3])
        a_y = int(self.action_threshold * self.bb[2])
        if(action == 0):
            #Horizontal - left
            self.bb[1] -= a_x
            if(self.bb[1] < 0):
                self.bb[1] = 0
        elif(action == 1):
            #Horizontal - right
            self.bb[1] += a_x
            if(self.bb[1] + self.bb[3] > IMG_WIDTH - 1):
                self.bb[1] = IMG_WIDTH - 1 - self.bb[3]
        elif(action == 2):
            #Vertical - Up
            self.bb[0] += a_y
            if(self.bb[0] + self.bb[2] > IMG_HEIGHT - 1):
                self.bb[0] = IMG_HEIGHT - 1 - self.bb[2]
        elif(action == 3):
            #Vertical - Down
            self.bb[0] -= a_y
            if(self.bb[0] < 0):
                self.bb[0] = 0
        elif(action == 4):
            #Scale - decrease
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 5):
            #Aspect ratio - fatter
            self.bb[0] += a_y
            self.bb[2] -= 2 * a_y
            if(self.bb[2] < MIN_HEIGHT):
                new_fac = (MIN_HEIGHT - self.bb[2]) / 2
                self.bb[0] -= new_fac
                self.bb[2] += 2 * new_fac
        elif(action == 6):
            #Aspect ratio - taller
            self.bb[1] += a_x
            self.bb[3] -= 2 * a_x
            if(self.bb[3] < MIN_WIDTH):
                new_fac = (MIN_WIDTH - self.bb[3]) / 2
                self.bb[1] -= new_fac
                self.bb[3] += 2 * new_fac
        elif(action == 7):
            #Trigger
            self.is_finished = True
        
        return old_bb, action, self.bb, self.is_finished

    def reset_bb(self):
        self.bb = init_bounding_box(self.full_env.shape, self.init_bb_threshold)
    
    def black_out(self):
        self.full_env = set_bb_to_black(self.full_env, self.bb)
    
    def extract_bound_box_image(self, bb):
        """
        extracts pixel content of current bounding box
        """
        bb_img = crop_pad_image(self.full_env, bb)
        return bb_img
       
