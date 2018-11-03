import os 
import sys
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models 
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.in_features = num_ftrs
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
class Bi_DQN(nn.Module):
    def __init__(self, n_action, global_feature_net, local_feature_net, num_fc_nodes, features_net_freeze_all = True, use_chexnet=False):
        super(Bi_DQN, self).__init__()
        
        #Freeze layers
        if(features_net_freeze_all):
            if(use_chexnet == False):
                for param in global_feature_net.parameters():
                    param.requires_grad = False
                for param in local_feature_net.parameters():
                    param.requires_grad = False
            else:
                for param in global_feature_net.densenet121.parameters():
                    param.requires_grad = False
                for param in local_feature_net.densenet121.parameters():
                    param.requires_grad = False
        
        if(use_chexnet == False):
            #Global feature net
            #Remove last layer
            num_glob_features = global_feature_net.fc.in_features
            global_feature_net.fc = nn.Linear(num_glob_features, num_fc_nodes)
            self.global_feature_net = global_feature_net
            #To hold global features for an environment to avoid multiple passing
            self.global_features = None
            #Local feature net
            #Remove last layer
            num_loc_features = local_feature_net.fc.in_features
            local_feature_net.fc = nn.Linear(num_glob_features, num_fc_nodes)
            self.local_feature_net = local_feature_net
        else:
            #Global feature net
            #Remove last layer
            num_glob_features = global_feature_net.in_features
            global_feature_net.densenet121.classifier =  nn.Sequential(nn.Linear(num_glob_features, num_fc_nodes))
            self.global_feature_net = global_feature_net
            self.global_features = None
            #Local feature net
            #Remove last layer
            num_loc_features = local_feature_net.in_features
            local_feature_net.densenet121.classifier =  nn.Sequential(nn.Linear(num_loc_features, num_fc_nodes))
            self.local_feature_net = local_feature_net
        

        #Fully connected layer1
        self.num_glob_features = num_glob_features
        self.num_loc_features = num_loc_features
        self.fc1 = nn.Linear(num_fc_nodes+num_fc_nodes, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, num_fc_nodes)
        self.output = nn.Linear(num_fc_nodes, n_action)
        self.leakyRelu = nn.LeakyReLU(0.01)
        
    def forward(self, full_env, bb_env):
        #Get global features
        if(self.global_features == None):
            global_features = self.global_feature_net(full_env)
        else:
            global_features = self.global_features
        #Get local features
        local_features = self.local_feature_net(bb_env)
        #Combine global and local features
        comb_features = torch.cat((global_features, local_features), 1)
        fc1_output = self.leakyRelu(self.fc1(comb_features))
        fc2_output = self.leakyRelu(self.fc2(fc1_output))
        #Softmax?
        output = self.output(fc2_output)
        return output

class DQN(nn.Module):
    def __init__(self, n_action, feature_net, num_fc_nodes, features_net_freeze_all = True, use_chexnet=False):
        super(DQN, self).__init__()
        
        #Freeze layers
        if(features_net_freeze_all):
            if(use_chexnet == False):
                for param in feature_net.parameters():
                    param.requires_grad = False
            else:
                for param in feature_net.densenet121.parameters():
                    param.requires_grad = False
                    
        if(use_chexnet == False):
            #Remove last layer
            #resnet
            num_features = feature_net.fc.in_features
            fc0 = nn.Linear(num_features, num_fc_nodes)
            torch.nn.init.xavier_uniform_(fc0.weight)
            feature_net.fc = fc0
            """
            #vgg
            num_features = feature_net.classifier[-1].in_features
            feature_net.classifier = nn.Sequential(nn.Linear(num_features, num_fc_nodes))
            """
            self.feature_net = feature_net
        else:
            #Global feature net
            #Remove last layer
            num_features = feature_net.in_features
            feature_net.densenet121.classifier =  nn.Sequential(nn.Linear(num_features, num_fc_nodes))
            self.feature_net = feature_net

        #Fully connected layer1
        self.num_features = num_features
        self.fc1 = nn.Linear(num_fc_nodes, num_fc_nodes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        #self.fc2 = nn.Linear(num_fc_nodes, num_fc_nodes)
        self.output = nn.Linear(num_fc_nodes, n_action)
        torch.nn.init.xavier_uniform_(self.output.weight)
        self.leakyRelu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        features = F.relu(self.feature_net(x))
        fc1_output = F.relu(self.fc1(features))
        #fc2_output = self.leakyRelu(self.fc2(fc1_output))
        #Softmax?
        output = self.output(fc1_output)
        return output
    
def get_vgg19_model(n_class, pretrained=True, num_freeze=7, checkpoint_path = None, trained_in_data_para = False):
    model_conv = models.vgg19(pretrained=pretrained)
    model_conv.features[36] = nn.AdaptiveAvgPool2d(1)
    #Update classifier
    model_conv.classifier = nn.Sequential(nn.Linear(512, 512, True),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(512, 512, True), nn.ReLU(inplace=True),nn.Dropout(p=0.5), nn.Linear(512, n_class, True))
    """
    # Number of filters in the bottleneck layer
    num_ftrs = model_conv.classifier[6].in_features
    print(num_ftrs)
    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]
    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, n_class)])
    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)
    """
    
    ct = 0
    for name, child in model_conv.named_children():
        ct += 1
        if ct < num_freeze:
            for name2, params in child.named_parameters():
                params.requires_grad = False
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if(trained_in_data_para):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if("module." in k):
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
            model_conv.load_state_dict(new_state_dict)
        else:
            model_conv.load_state_dict(checkpoint)
    return model_conv
            
def get_resnet50_model(n_class, pretrained=True, num_freeze=7, checkpoint_path = None, trained_in_data_para = False):
    model_conv = models.resnet50(pretrained=pretrained)
    model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_class)
    ct = 0
    for name, child in model_conv.named_children():
        ct += 1
        if ct < num_freeze:
            for name2, params in child.named_parameters():
                params.requires_grad = False
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if(trained_in_data_para):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if("module." in k):
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
            model_conv.load_state_dict(new_state_dict)
        else:
            model_conv.load_state_dict(checkpoint)
    return model_conv

def get_chexnet(n_class, pretrained=True, checkpoint_path=None):
    model = DenseNet121(n_class)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
        state_dict = checkpoint['state_dict']
        remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[key]
        
        model.load_state_dict(checkpoint['state_dict'])
    return model
