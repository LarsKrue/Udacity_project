# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:28:12 2020

- infer a category from an image
- relaod pretrained model and specifically trained classifier

@author: Lars Krüger
"""

import torch as tr

import image_functions as imf # own image functions
import network_functions as nf # own network functions

from get_input import get_predict_input # command Line arguments




# get user input 
input_predict_args = get_predict_input()
configuration_path = input_predict_args.config
state_dict_path = input_predict_args.state
print(input_predict_args)


# load pretrained NN-model
configuration = tr.load(configuration_path)
state_dict = tr.load(state_dict_path)
my_reloaded_network = nf.reload_network_predict(configuration)


# overview pretrained own classifier
print("------------------------Inference-------------------------------------")
print("No of input units: {}".format(configuration["input_units"]))
print("No of hidden units: {}".format(configuration["hidden_units"]))
print("Architecture of NN: {}".format(configuration["architecture"]))
print("Accuracy on validation data set: {}".format(configuration["accuracy"]))
print("Epochs trained: {}".format(configuration["epochs"]))
print("Trained on: {}".format(configuration["device"]))
tot_time = configuration["total_time"]
print("Time needed to train: ", str(int((tot_time/3600)))+"h:"+str(int((tot_time%3600)/60))+"min:"
      +str(int((tot_time%3600)%60))+"sek" )

# setup and re-load pretrained classifier
my_reloaded_network = nf.reload_classifier(configuration,state_dict, my_reloaded_network)

# PIL image Loading
test_image = imf.img_load(input_predict_args)

# image preprocessing PIL to Tensor
img = imf.image_preprocess(input_predict_args)

# inference and top5 results
my_reloaded_network.eval() # inference mode
k=input_predict_args.k #k-top results from input parser / CL

device = nf.check_cuda_availability(input_predict_args)

with tr.no_grad(): # no gradient calculation
    my_reloaded_network.to(device)
    img=img.to(device)
    logps = my_reloaded_network.forward(img)
    ps = tr.exp(logps)
    top_p, top_class = ps.topk(k, dim=1, sorted=True)

if device == "cuda:0":
    img = img.cpu()
    my_reloaded_network = my_reloaded_network.to("cpu")

# prepare inference results for label name matching
top_label_list = top_class.tolist()[0] # tensor to list, first element got get dim 1 list
top_p_list = top_p.tolist()[0] # tensor to list, first element got get dim 1 list
top_p_list = [100 * p for p in top_p_list] # get percent values (%)

# load json file, create dict
cat_no_to_flower_name_dict = imf.load_flower_cat_names(input_predict_args)

# map category names to top5 labels
category_names_list = [cat_no_to_flower_name_dict[str(label)] for label in top_label_list]

# combination of top-k results category number and flower name as result
result_list=[str((names, labels)) for names, labels in zip(top_label_list,category_names_list)]

print("\n----------------------Prediction results-----------------------------\n")
for i in range(k): 
    print("Pos. {}: Category: {} - Probability: {}% ".format(i+1, result_list[i], round(top_p_list[i],2)))
  
    
# plot img + Category names & probability
imf.plot_prediction(configuration,input_predict_args, result_list, top_p_list, test_image, img)

