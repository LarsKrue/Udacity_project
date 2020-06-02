# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:09:49 2020

- helper function module for setup and (re)laod of neural networks

@author: Lars Krüger
"""

import torch as tr

from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
from collections import OrderedDict



#---------------------------------Load network / two options------------------
def load_network(in_args):
        
    if in_args.arch == "densenet121":
        my_network = models.densenet121(pretrained=True)
        input_units_nn = 1024 # per definition
        print("\n----------------loading pretrained model-----------------\n")
        #print(summary(my_network, (3,224,224)))
        print("Densenet121 pretrained model with {} input units loaded\n".format(input_units_nn))
    
    else:
        my_network = models.vgg19(pretrained=True)
        input_units_nn = 25088 # per definition
        print("\n----------------loading pretrained model-----------------\n")
        #print(summary(my_network, (3,224,224)))
        print("VGG19 pretrained model with {} input units loaded\n".format(input_units_nn))
    return(my_network, input_units_nn)

#--------------------------Reload network model structure ---------------------

def reload_network_predict(configuration):
    
    if configuration["architecture"] == "densenet121":
           my_reloaded_network = models.densenet121(pretrained=True)
           print("Densenet121 pretrained model loaded")
           
    else:
           my_reloaded_network = models.vgg19(pretrained=True)
           print("VGG19 pretrained model loaded")
          
    
    return my_reloaded_network


#--------------------------Reload trained network model classifier-------------

def reload_classifier(configuration,state_dict, my_network):

    input_units = configuration["input_units"]
    hidden_units = configuration["hidden_units"]
    output_units = configuration["output_units"]
    
    
    
    classifier = nn.Sequential(OrderedDict([
                              ("Layer 1", nn.Linear(input_units, hidden_units)),
                              ("ReLu", nn.ReLU()),
                              ("Layer 2", nn.Linear(hidden_units, output_units)),
                              ("Output", nn.LogSoftmax(dim=1))
                              ]))
    
    #Assignment of new classifier structure to network, but yet untrained
    my_network.classifier = classifier
    my_network.load_state_dict(state_dict, strict=False) # pytorch forum, otherwise missing & unexpected keys
    
    return my_network



#---------------Setup Image loader for train, test and validation--------------

def setup_loaders(in_args):
    #----------------create data loaders for --------

    size = 250
    crop = 224
    degrees = 30
    
    # data augmentation through random application of radom transformation, only one picked randomly
    random_list = [transforms.RandomHorizontalFlip(), transforms.RandomPerspective(),transforms.RandomRotation(degrees)]
    
    train_transform = transforms.Compose([transforms.Resize((size, size)),# returns PIL image
                                           transforms.CenterCrop(crop),# returns PIL image
                                           transforms.RandomChoice(random_list),# returns PIL image
                                           transforms.ToTensor(), # returns tensor / expects PIL image
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # expects tensor                                
                                           ])
    
    # test and validation data transforms - resizing, cropping, and normalizing according to standard inputs of pretrained nns
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.CenterCrop(crop),
                                          transforms.ToTensor(), # returns tensor / expects PIL image
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                         ])
    
    vali_transform = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.CenterCrop(crop),
                                          transforms.ToTensor(), # returns tensor / expects PIL image
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                         ])
    # data laoders for each data set
    data_dir = in_args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load the training data into loader object
    batch_size = 32
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = tr.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Load the test data into loader object
    test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = tr.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Load the vali data into loader object
    vali_set = datasets.ImageFolder(valid_dir, transform=vali_transform)
    vali_loader = tr.utils.data.DataLoader(vali_set, batch_size=batch_size, shuffle=False)

    return(train_loader, test_loader, vali_loader, batch_size)


#-------------------------------------New Classifier for pretrained NN---------

def setup_new_classifier(my_network,input_units_nn, in_args):
   

    # Freeze paramters so backpropagation wont update them 
    for paramet in my_network.parameters():
        paramet.requires_grad = False
        
    
    # Design of new Classifer Network
    input_units = input_units_nn # input of original classifier 
    hidden_units = in_args.hidden_units # only one hidden layer
    output_units = 102  #number of flower categories
    
    
    classifier = nn.Sequential(OrderedDict([
                              ("Layer 1", nn.Linear(input_units, hidden_units)),
                              ("ReLu", nn.ReLU()),
                              ("DropOut1",nn.Dropout(p=0.5)),
                              ("Layer 2", nn.Linear(hidden_units, output_units)),
                              ("Output", nn.LogSoftmax(dim=1))
                              ]))
    
    #Assignment of new classifeir steup to network, but yet untrained
    my_network.classifier = classifier #classifier
    
    
    # Define loss function called criterion
    criterion = nn.NLLLoss() # 
    
    # OTrain only classifier
    optimizer = optim.Adam(my_network.classifier.parameters(), lr=in_args.learning_rate)
    
    return my_network, criterion, optimizer, input_units, output_units, hidden_units

#-------------------------------------CUDA-------------------------------------

def check_cuda_availability(in_args):
    device = tr.device("cuda:0" if (tr.cuda.is_available() and in_args.device_type == "gpu") else "cpu")
    print("\n-----------------------------------------------------------------")
    print("Device used for computations: {}".format(device))  

    return device

#---------------------------------Run time------------------------------------
    
def runtime(tot_time):
    print("\n** Total Elapsed Runtime:",
      str(int((tot_time/3600)))+"h:"+str(int((tot_time%3600)/60))+"min:"
      +str(int((tot_time%3600)%60))+"sek" )
    return None
