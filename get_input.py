"""
Created on Fri May 29 12:41:15 2020

input: -

output: Command Line Arguments and respectrive defauls values for train.py 
        flower image classifier adapted network
        Udacity final project Intro AI programming
@author: Lars Krüger
"""


import argparse

def get_input():   
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Parsen von 4 User-Inputs", prog="Train")
    #  5 command line arguments modell architecture / hyperparameters
    parser.add_argument('--dir', type = str, default='flower_data/', 
                        help='path to folder of flower images')
    parser.add_argument('--arch',type = str, default = 'VGG16', #vgg19, resnet, alexnet 
                        help='NN Architecture - VGG16, VGG19')
    parser.add_argument('--epochs',type = int, default = 3,  
                        help='Number of training epochs')
    parser.add_argument('--learning_rate',type = float, default = 0.03,  
                        help='Learning Rate')
    parser.add_argument('--hidden_units',type = int, default = 4096,  
                        help='no of hidden units in classifier layer')
    parser.add_argument('--Device_type',type = str, default = 'cpu',  
                        help='Usage of GPUs')
    in_arg = parser.parse_args() 
    return (in_arg)
