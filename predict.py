# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:28:12 2020

- infer a category from an image
- pretrained model and specifically trained classifier

@author: Lars Krüger
"""
import matplotlib.pyplot as plt
import torch as tr
import os
import json

from get_input import get_predict_input # command Line arguments
from torchvision import  transforms, models
from torch import nn
from collections import OrderedDict
from skimage import io
from PIL import Image


def load_flower_cat_names(input_predict_args):  
    #---------------------Load category names from .json file ----------------
    # "cat_to_name.json" in directory flower_data
    
    dir_name = input_predict_args.dir
    file_name = "cat_to_name.json"
    file = os.path.join(dir_name, file_name)
    with open(file, "r") as f:
        cat_no_to_flower_name_dict = json.load(f)
    #print("Network to be trained on {} categories".format(len(cat_no_to_flower_name_dict.keys())))
    return(cat_no_to_flower_name_dict)

def map_label_numbers_to_cat_names(label_list, cat_no_to_flower_name_dict):
    # --------------------- Mapping cat numbers to flower names----------------
    category_names = []
    for label in label_list:
        category_names = category_names.append(cat_no_to_flower_name_dict[str(label)]) # key in dict is string/ category is int
    print("Liste: ", category_names)
    return category_names


#---------------get user input------------------------------------------------

input_predict_args = get_predict_input()
print(input_predict_args)

#--------------- load pretrained NN and pretrained classifier-----------------

configuration = tr.load("configuration_dict.pth")


if configuration["architecture"] == "resnet18":
       my_network = models.resnet18(pretrained=True)
       print("Resnet18 pretrained model loaded")
       input_units_nn = 512 # per definition
else:
       my_network = models.resnet50(pretrained=True)
       print("Resnet50 pretrained model loaded")
       input_units_nn = 2048 # per definition

#-----------------------------Load pretrained own classifier ------------------


# Overview over trained classifier
print("No of input units: {}".format(configuration["input_units"]))
print("No of hidden units: {}".format(configuration["hidden_units"]))
print("Architecture of NN: {}".format(configuration["architecture"]))
print("Accuracy on validation data set: {}".format(configuration["accuracy"]))
print("Epochs trained: {}".format(configuration["epochs"]))
print("Trained on: {}".format(configuration["device"]))
tot_time = configuration["total_time"]
print("Time needed to train: ", str(int((tot_time/3600)))+"h:"+str(int((tot_time%3600)/60))+":min"
      +str(int((tot_time%3600)%60))+":sek" )

#-- ensure compatibility with specifically created new classifier 
# Design of new Classifer Network
input_units = configuration["input_units"]
hidden_units = configuration["hidden_units"]
output_units = configuration["output_units"]

state_dict = tr.load("checkpoint_trained_network.pth")
my_network.load_state_dict(state_dict, strict=False) # pytorch forum, otherwise missing & unexpected keys


classifier = nn.Sequential(OrderedDict([
                          ("Layer 1", nn.Linear(input_units, hidden_units)),
                          ("ReLu", nn.ReLU()),
                          ("Layer 2", nn.Linear(hidden_units, output_units)),
                          ("Output", nn.LogSoftmax(dim=1))
                          ]))

#Assignment of new classifier structure to network, but yet untrained
my_network.fc = classifier

#---------------------Image Loading-------------------------------------------

file_name =input_predict_args.file

test_image = io.imread(file_name)
#test_image = test_image[250:850, 250:650] # manual center crop
plt.imshow(test_image) # all channels, real image

print(test_image.size)
print(test_image.shape) #3 channel image, RGB

#--------------------Image preprocessing--------------------------------------

size = 250
crop = 224
single_transform = transforms.Compose([transforms.Resize((size, size)),# returns PIL image
                                       transforms.CenterCrop(crop),
                                       transforms.ToTensor(), # returns tensor
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # expects tensor                                
                                       ])
img = Image.open(file_name)
img = single_transform(img) # 
img = img.unsqueeze(0)

#--------------------------Inference------------------------------------------
my_network.eval() # inference mode
k=5 #k-top results
with tr.no_grad(): # no gradient calculation
    logps = my_network.forward(img)
    ps = tr.exp(logps)
    print("ps", ps)
    top_p, top_class = ps.topk(k, dim=1)
    print("Top p", top_p)
    print("Top Class", top_class)
    print("Tuples:", zip(top_p, top_class))
 


#----------------------------Map label numbers to category names--------------
top_label_list = top_class.tolist()[0] # tensor to list, first element got get dim 1 list
top_p_list = top_p.tolist()[0] # tensor to list, first element got get dim 1 list
top_p_list = [100 * p for p in top_p_list]
print("Top Class:",top_label_list)
print("Top prob:",top_p_list)

# load json file
cat_no_to_flower_name_dict = load_flower_cat_names(input_predict_args)
# mapp category names to top5 labels
category_names_list = [cat_no_to_flower_name_dict[str(label)] for label in top_label_list ]
print(category_names_list)


#-----------------------------Plot img + Category Probability------------------

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,6), ncols=3)
ax1.barh([1,2,3,4,5],top_p_list,0.6,tick_label=category_names_list)

ax1.set_aspect("auto")
ax1.set_title('Class Probability')
ax2.imshow(img.resize_(1, 224, 224).numpy().squeeze(), aspect=1)
ax2.axis('on')
ax2.set_title(file_name + " - processed") 
ax3.imshow(test_image)
ax3.axis('on')
ax3.set_title(file_name + " - real") 
