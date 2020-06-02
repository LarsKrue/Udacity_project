# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:41:15 2020

input: -

output: Command Line Arguments and respectrive defauls values for train.py & predict.py
        flower image classifier adapted network
        Udacity final project
        Intro AI programming


@author: Lars Krüger
"""
import argparse

def get_input():   
    # Create Parse using ArgumentParser for train.py
    parser = argparse.ArgumentParser(description="Parsen von 4 User-Inputs", prog="Train")
    #  5 command line arguments modell architecture / hyperparameters
    parser.add_argument('--dir', type = str, default='flower_data', 
                        help='path to folder of flower images')
    parser.add_argument('--arch',type = str, default = 'densenet121', #or VGG19, 
                        help='NN Architecture - densenet121 or vgg19')
    parser.add_argument('--epochs',type = int, default = 2,  
                        help='Number of training epochs')
    parser.add_argument('--learning_rate',type = float, default = 0.001,  
                        help='Learning Rate')
    parser.add_argument('--hidden_units',type = int, default = 4096,  
                        help='no of hidden units in classifier layer')
    parser.add_argument('--device_type',type = str, default = 'gpu',  #  'cpu'
                        help='Usage of GPUs')
    parser.add_argument('--checkpoint_dir',type = str, default = 'checkpoint_trained_network.pth',  #  'cpu'
                        help='path + file of state_dict')
    in_arg = parser.parse_args() 
    return (in_arg)

def get_predict_input():   
    # Create Parse using ArgumentParser for predict.py
    parser_predict = argparse.ArgumentParser(description="Parsen von User-Inputs", prog="Predict")
    #  command line arguments modell architecture / hyperparameters
    parser_predict.add_argument('--file', type = str, default='image_12.jpg', 
                        help='path to folder and name of image')
    parser_predict.add_argument('--dir', type = str, default='flower_data', 
                        help='path to folder of flower images')
    parser_predict.add_argument('--config', type = str, default='configuration_dict.pth', 
                        help='path to configuration file')
    parser_predict.add_argument('--state', type = str, default="checkpoint_trained_network.pth", 
                        help='path to state_dict')
    parser_predict.add_argument('--k', type = int, default=5, 
                        help='k-top results')
    parser_predict.add_argument('--device_type',type = str, default = 'gpu',  #  'cpu'
                        help='Usage of GPUs')
    parser_predict.add_argument('--cat_names',type = str, default = 'cat_to_name.json',  
                        help='Dir and filename of category names')
    in_predict_arg = parser_predict.parse_args() 
    return (in_predict_arg)
