# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:45:04 2020

Classifier Project Udacity 

train classifier of a pretrained torchvision model


@author: Lars Krüger
"""
import time

import network_functions as nf # own network functions
import image_functions as imf # own image functions
import torch as tr

from get_input import get_input # command Line arguments



#---------------------Start Time-----------------------------------------------
start_time = time.time() 

#---------------------Load input from parser get_input() / default values ----

# defauls:
# epochs:2
# devices: CUDA
# architecture: Densenet121
# dir_data: flower_data/
# learning rate: 0.001
# hidden units: 4096

in_args=get_input()
print("------------------------Training--------------------------------------")
print("- Train {}-Network with {} hidden units on device: {}".format(in_args.arch,in_args.hidden_units,in_args.device_type))
print("- {} Training epochs with learning rate of {}".format(in_args.epochs,in_args.learning_rate))
print("- Image files in {}".format(in_args.dir))



#----------------create data loaders for tran, test and validation-------------
train_loader, test_loader, vali_loader, batch_size = nf.setup_loaders(in_args)


#--------------------Select and load pretrained NN --------------------------- 
my_network, input_units_nn = nf.load_network(in_args)
#print(my_network)


#--------------------Setup new Classifier / criterion, optimizer  NN ----------
my_network, criterion, optimizer,input_units, output_units, hidden_units = nf.setup_new_classifier(my_network,input_units_nn, in_args)



#---------Check for CUDA availability and assign computation device------------
device = nf.check_cuda_availability(in_args)


#-------------------Training classifier---------------------------------------
epochs = in_args.epochs
train_losses, test_losses, accuracy_list = [], [], []

# Move to CUDA only for forward & backpropagation
my_network=my_network.to(device) 

for loop in range(epochs):
    running_loss_train, running_loss_test  = 0, 0
    print("Start Training")
 
    #-----------------------Training-Forward Pass------------------------------
   
    for images, labels in train_loader:
  
        # Move data to cuda if available
        images = images.to(device)
        labels = labels.to(device)
        
        # clear gradients
        optimizer.zero_grad()
                
        # Probability Distribution  / output
        logps = my_network.forward(images)
        
        #Loss Function
        loss = criterion(logps, labels)
        running_loss_train += loss.item()
        
        # Backward pass / Back Propagation
        loss.backward()
        
        #Update Weights
        optimizer.step()
        
        
    else: # for else loop, if no break / loop terminated normally, then Else
        
        # CUDA cache economy 
        if device == "cuda:0":
            images = images.to("cpu")
            labels = labels.to("cpu")
        accuracy = 0
        print("Start Test")
        #-----------------------Test Cycle-----------------------------------
        
        # turn-off gradient calculation for validation, saves memory + computations
        with tr.no_grad():
            for images, labels in test_loader:
                
                # Move data to cuda if available
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward Loop
                logps = my_network(images)
                running_loss_test += criterion(logps, labels)
                
                
                # accuracy calculations
                ps = tr.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape) # True where identity
                accuracy += tr.mean(equals.type(tr.FloatTensor))
        # CUDA cache economy 
        if device == "cuda:0":
            images = images.to("cpu")
            labels = labels.to("cpu") 
        
        print("After Epoch {} of {}: ".format(loop+1, epochs))
        print("Avg. Training Loss: {}".format(running_loss_train/len(train_loader)))
        print("Avg. Test Loss: {}".format(running_loss_test/len(test_loader)))
        print("Avg. Testing Accuracy: {}".format(accuracy/len(test_loader)))
        print("---------------------------------------------------------------")
        train_losses.append(running_loss_train) # for plotting
        test_losses.append(running_loss_test) #for plotting
        accuracy_list.append((accuracy/len(test_loader))) #for plotting



#---------------------------- validation--------------------------------------
# turn-off gradient calculation for validation, saves memory + computations
accuracy_val = 0
vali_losses = 0
k = 1 # k-top categories

with tr.no_grad():
    for images, labels in vali_loader:
        
        # Move data to cuda if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward Loop
        logps = my_network(images)
        vali_losses += criterion(logps, labels)
        
        
        # accuracy calculations
        ps = tr.exp(logps)
        top_p, top_class = ps.topk(k, dim=1)
        equals = top_class == labels.view(*top_class.shape) # True where identity
        accuracy_val += tr.mean(equals.type(tr.FloatTensor))

x = (accuracy_val/len(vali_loader)).item() # as of type tensor
print("Accuracy on validation data set: ", x)

#---------------------------------return to CPU-------------------------------

# CUDA cache economy 
if device == "cuda:0":
    my_network = my_network.to("cpu") 
    images = images.to("cpu")
    labels = labels.to("cpu") 

#---------------------------------Runtime calculations-------------------------
end_time = time.time()
tot_time = end_time - start_time
print(nf.runtime(tot_time))


# --------------------------------Save checkpoint-----------------------------
# two files otherwise key errors in state_dict

state_dict_file = in_args.checkpoint_dir
checkpoint = my_network.state_dict()
tr.save(checkpoint, state_dict_file)

#---------------------------Save config incl total time----------------------- 
# all architecture information needed!
configuration_dict = {"input_units": input_units,
              "output_units" : output_units,
              "hidden_units" : hidden_units,
              "architecture" : in_args.arch,
              "accuracy" : x,
              "epochs" : epochs,
              "device": device,
              "total_time": tot_time              
              }
tr.save(configuration_dict, "configuration_dict.pth")


#-----------------------Plot Training & Testing-------------------------------

imf.plot_performance(in_args, train_losses, test_losses, accuracy_list)

