Classifier Project Udacity 

train classifier of a pretrained torchvision model
resnet18 or 50

@author: Lars Krüger
"""
import time


import torch as tr
from torch import nn
import time
import numpy as np

from torchvision import datasets, transforms, models
from torch import optim
from get_input import get_input # command Line arguments
from collections import OrderedDict


    
def setup_loaders(in_args):
    #----------------create data loaders for tran, test and validation---------

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
    batch_size = 64
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = tr.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Load the test data into loader object
    test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = tr.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Load the vali data into loader object
    vali_set = datasets.ImageFolder(valid_dir, transform=vali_transform)
    vali_loader = tr.utils.data.DataLoader(vali_set, batch_size=batch_size, shuffle=False)

    return(train_loader, test_loader, vali_loader, batch_size)
 
def load_network(in_args):
        
    if in_args.arch == "resnet18":
        my_network = models.resnet18(pretrained=True)
        input_units_nn = 512 # per definition
        print("Resnet18 pretrained model with {} input units loaded".format(input_units_nn))
        
    
    else:
        my_network = models.resnet50(pretrained=True)
        input_units_nn = 2048 # per definition
        print("Resnet18 pretrained model with {} input units loaded".format(input_units_nn))
    return(my_network, input_units_nn)

def setup_new_classifier(my_network,input_units_nn, in_args):
    #--------------------Setup new classifier to pretrained NN -------------------

    # Freeze paramters so backpropagation wont update them 
    for paramet in my_network.parameters():
        paramet.requires_grad = False
        
    
    # Design of new Classifer Network
    input_units = input_units_nn # input of original classifier / final layer resnet 18 / 50 classifiers
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
    my_network.fc = classifier
    
    
    # Define loss function called criterion
    criterion = nn.NLLLoss() # 
    
    # OTrain only classifier
    optimizer = optim.Adam(my_network.fc.parameters(), lr=in_args.learning_rate)
    
    return my_network, criterion, optimizer, input_units, output_units, hidden_units

def check_cuda_availability(in_args):
    device = tr.device("cuda:0" if (tr.cuda.is_available() and in_args.Device_type == "cuda") else "cpu")
    print("Device used for computations: {}".format(device))

    return device


#-----------------------------------------------------------------------------


#---------------------Load input from parser get_input() / default values ----

# defauls:
# epochs:3
# devices: CUDA
# learning rate 0.03
# architecture: VGG16
# dir_data: flower_data/
# learning rate: 0.03

in_args=get_input()
print("- Train {}-Network with {} hidden units on device: {}".format(in_args.arch,in_args.hidden_units,in_args.Device_type))
print("- {} Training epochs with learning rate of {}".format(in_args.epochs,in_args.learning_rate))
print("- Image files in {}".format(in_args.dir))



#----------------create data loaders for tran, test and validation-------------
train_loader, test_loader, vali_loader, batch_size = setup_loaders(in_args)


#--------------------Select and load pretrained NN --------------------------- 
my_network, input_units_nn = load_network(in_args)
print(my_network)


#--------------------Setup new Classifier / criterion, optimizer  NN ----------
my_network, criterion, optimizer,input_units, output_units, hidden_units = setup_new_classifier(my_network,input_units_nn, in_args)



#---------Check for CUDA availability and assign computation device------------
device = check_cuda_availability(in_args)


#-------------------Training classifier---------------------------------------
# Training classifier
epochs = in_args.epochs

train_losses, test_losses, accuracy_list = [], [], []

# CUDA only for forward & backpropagation
# Move network to Cuda
my_network.to(device) 

for loop in range(epochs):
    running_loss = 0
    print("Start Training")
    start_time = time.time() 
   
    #-----------------------Training-Forward Pass------------------------------
   
    for images, labels in train_loader:
        
        # Move data to cuda if available
        images = images.to(device)
        labels = labels.to(device)
        
        # clear gradients
        optimizer.zero_grad()
                
        # Probability Distribution  / output
        logps = my_network.forward(images)
        
        # adjustment
        #logps = tr.squeeze(logps)
        #labels = labels.float()
        
        
        #Loss Function
        loss = criterion(logps, labels)
        running_loss += loss.item()
        
        # Backward pass / Back Propagation
        loss.backward()
        
        #Update Weights
        optimizer.step()
        
    else: # for else loop, if no break / loop terminated normally, then Else
        test_loss = 0
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
                test_loss += criterion(logps, labels)
                
                
                # accuracy calculations
                ps = tr.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape) # True where identity
                accuracy += tr.mean(equals.type(tr.FloatTensor)).item()
                
        train_losses.append(running_loss/len(train_loader))# batch size?
        test_losses.append(test_loss/len(train_loader))
        accuracy_list.append(accuracy)
        
        print("After Epoch {} of {}".format(loop+1, epochs))
        print("Avg. Training Error {}".format(running_loss))
        print("Avg. Test Error {}".format(test_loss))
        print("Avg. Test Accuracy {}".format(accuracy))


# --------------------------------save checkpoint-----------------------------
# two files otherwise key errors in state_dict

checkpoint = my_network.state_dict()
tr.save(checkpoint, "checkpoint_trained_network.pth")

#---------------------------- validation--------------------------------------
# turn-off gradient calculation for validation, saves memory + computations
accuracy_val = 0
k = 1 # k-top categories
with tr.no_grad():
    for images, labels in vali_loader:
        
        # Move data to cuda if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward Loop
        logps = my_network(images)
        test_loss += criterion(logps, labels)
        
        
        # accuracy calculations
        ps = tr.exp(logps)
        top_p, top_class = ps.topk(k, dim=1)
        equals = top_class == labels.view(*top_class.shape) # True where identity
        accuracy_val += tr.mean(equals.type(tr.FloatTensor)).item()
print("Accuracy on validation data set".format(accuracy_val))


#---------------------------------return to CPU-------------------------------
if device == "cuda":

    my_network = my_network.tp("cpu")
    images = images.to("cpu")
    labels = labels.to("cpu")


#---------------------------------time stop-----------------------------------
end_time = time.time()
tot_time = end_time - start_time
print("\n** Total Elapsed Runtime:",
      str(int((tot_time/3600)))+"h:"+str(int((tot_time%3600)/60))+":min"
      +str(int((tot_time%3600)%60))+":sek" )

#---------------------------Save config incl total time----------------------- 
# all architecture information needed!
configuration_dict = {"input_units": input_units,
              "output_units" : output_units,
              "hidden_units" : hidden_units,
              "architecture" : in_args.arch,
              "accuracy" : accuracy,
              "epochs" :epochs,
              "device": device,
              "total_time": tot_time
              }
tr.save(configuration_dict, "configuration_dict.pth")
