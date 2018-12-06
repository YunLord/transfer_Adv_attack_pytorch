# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:10:15 2018

@author: Yun
"""

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import SiameseNetworkDataset
from SiameseNetwork import SiameseNetwork,ContrastiveLoss

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

# Hyper-parameters
param = {
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}
    
class Config():
#    # Data loaders
#    train_dataset = datasets.MNIST(root='../data/',train=True, download=True, 
#        transform=transforms.ToTensor())
#    loader_train = torch.utils.data.DataLoader(train_dataset, 
#        batch_size=param['batch_size'], shuffle=True)
#    
#    test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
#        transform=transforms.ToTensor())
#    loader_test = torch.utils.data.DataLoader(test_dataset,  
#        batch_size=param['test_batch_size'], shuffle=True)
#    train_batch_size = 64
#    train_number_epochs = 100
    training_dir = "data/faces/training/"
    testing_dir = "data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100
    is_gpu=False
    
def main():
    folder_dataset = datasets.ImageFolder(root=Config.training_dir)         
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)   
    

    vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=8)
    dataiter = iter(vis_dataloader)
    
    
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())
    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=Config.train_batch_size)
    if Config.is_gpu:
        net = SiameseNetwork().cuda()
    else:
        net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    counter = []
    loss_history = [] 
    iteration_number= 0
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            if Config.is_gpu:
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history)
    
if __name__ == '__main__':
    main()