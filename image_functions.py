# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:10:58 2020

- helper function module for image and category handling for NN training and inference

@author: Lars Krüger
"""

import matplotlib.pyplot as plt
import os
import json

from skimage import io
from PIL import Image
from torchvision import transforms


#----------------------------laod json file cat names-------------------------

def load_flower_cat_names(input_predict_args):  
    #---------------------Load category names from .json file ----------------
    # "cat_to_name.json" in directory flower_data
    
    dir_name = input_predict_args.dir
    file_name = input_predict_args.cat_names
    file = os.path.join(dir_name, file_name)
    with open(file, "r") as f:
        cat_no_to_flower_name_dict = json.load(f)    
    return(cat_no_to_flower_name_dict)



#------------------------plot training performance----------------------------
def plot_performance(in_args, train_losses, test_losses, accuracy_list):
    arch = in_args.arch
    epochs = in_args.epochs
    
    fig, (ax1, ax3) = plt.subplots(2, figsize=(12,8))
    fig.suptitle("Performance Network Training - {} ".format(arch))
    
    ax1.set_xlabel("After Epoch")
    plt.xticks(range(epochs),range(1,epochs+1))
    ax1.set_ylabel("Training Loss")
    ax1.grid()
    ax1.plot(train_losses, color="b", label="Training Loss")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Loss")
    ax2.plot(test_losses,color="r", label="Validation Loss")
    
    # one legend for two y axis
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    
    ax3.grid()
    ax3.set_xlabel("After Epoch")
    plt.xticks(range(epochs),range(1,epochs+1))
    
    ax3.set_ylabel("Accuracy [%]")
    accuracy_percent =[accuracy * 100  for accuracy in accuracy_list]
    ax3.plot(accuracy_percent, label='Test Accuracy')
    ax3.legend(frameon=True)
    plt.tight_layout()
    plt.show()
    
    return None

#---------------------------------- Load PIL- Image --------------------------
def img_load(input_predict_args):
    file_name =input_predict_args.file
    test_image = io.imread(file_name)
    #test_image = test_image[250:850, 250:650] # manual center crop
    #plt.imshow(test_image) # all channels, real image
    return test_image


#------------------------Image tensor preprocessing for prediction-------------
def image_preprocess(input_predict_args):
    file_name = input_predict_args.file
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
    return img


#-----------------------------Plot prediction---------------------------------
    
def plot_prediction(configuration,input_predict_args, result_list, top_p_list, test_image, img):
    file_name = input_predict_args.file
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,6), ncols=3)
    fig.suptitle("Prediction - Network {} - File {} ".format(configuration["architecture"], file_name))
    top_p_list = top_p_list[::-1] #reverse order to show top category in top position in plot
    result_list = result_list[::-1]
    ax1.barh(range(len(result_list)),top_p_list,0.6,tick_label=result_list)
    
    ax1.set_aspect("auto")
    ax1.grid()
    ax1.set_title('Class Probability [%]')
    ax2.imshow(img.resize_(1, 224, 224).cpu().numpy().squeeze(), aspect=1)
    ax2.axis('on')
    ax2.set_title(file_name + " - processed") 
    ax3.imshow(test_image)
    ax3.axis('on')
    ax3.set_title(file_name + " - real")
    return None
