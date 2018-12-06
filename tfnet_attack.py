# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:46:08 2018

@author: Yun
"""

from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import os  
from torchvision import datasets, models, transforms  

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data

from models import LeNet5


# Hyper-parameters
param = {
    'test_batch_size': 100,
    'epsilon': 0.3,
}

data_transforms = {  
    'train': transforms.Compose([  
        transforms.RandomSizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),  
    'val': transforms.Compose([  
        transforms.Scale(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),  
}  

# Data loaders
#test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
#    transform=transforms.ToTensor())
#loader_test = torch.utils.data.DataLoader(test_dataset, 
#    batch_size=param['test_batch_size'], shuffle=False)

data_dir = 'data/catdogdata'  
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  
         for x in ['train', 'val']}  
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=16,  
                                           shuffle=True, num_workers=0)
                for x in ['train', 'val']}   
# Setup model to be attacked
net = models.resnet18(pretrained=False)
dim_in = net.fc.in_features
net.fc = nn.Linear(dim_in, 2)

num_epoch = 20
for epoch in range(num_epoch):
    print('{}/{}'.format(epoch + 1, num_epoch))
    print('-' * 10)    
    net.load_state_dict(torch.load('model_save/'+str(epoch)+'.pth'))
    
    if torch.cuda.is_available():
        print('CUDA ensabled.')
        net.cuda()
    
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    
    test(net, dset_loaders['val'])
    
    
    # Adversarial attack
    adversary = FGSMAttack(net, param['epsilon'])
    # adversary = LinfPGDAttack(net, random_start=False)
    
    
    t0 = time()
    attack_over_test_data(net, adversary, param, dset_loaders['val'])
    print('{}s eclipsed.'.format(time()-t0))
    print('Finish attacking!')
    print()