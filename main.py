# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:20:34 2018

@author: Yun
"""
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.autograd import Variable  
import numpy as np  
import torchvision  
from torchvision import datasets, models, transforms  
import matplotlib.pyplot as plt  
import time  
import copy  
import os  

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

def imshow(inp, title=None):  
        """Imshow for Tensor."""  
        inp = inp.numpy().transpose((1, 2, 0))  
        mean = np.array([0.485, 0.456, 0.406])  
        std = np.array([0.229, 0.224, 0.225])  
        inp = std * inp + mean  
        plt.imshow(inp)  
        if title is not None:  
            plt.title(title)  
        plt.pause(0.001)  # pause a bit so that plots are updated  
        
def main():
    data_dir = 'catdogdata'  
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  
             for x in ['train', 'val']}  
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,  
                                                   shuffle=True, num_workers=0)  
                    for x in ['train', 'val']}  
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}  
    dset_classes = dsets['train'].classes  
    use_gpu = False
    fix_param = False    
    model_ft = models.resnet18(pretrained=True)  
    num_ftrs = model_ft.fc.in_features  
    model_ft.fc = nn.Linear(num_ftrs, 2)  
 #    print(model)
    if use_gpu:
        model_ft = model_ft.cuda()
    # define optimize function and loss function
    if fix_param:
        optimizer = optim.Adam(model_ft.fc.parameters(), lr=1e-4,weight_decay=0.00001)
    else:
        optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
        # train
    num_epoch = 20
    for epoch in range(num_epoch):
        print('{}/{}'.format(epoch + 1, num_epoch))
        print('-' * 10)
        print('--Train--')
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        since = time.time()
        for i,data in enumerate(dset_loaders['train'],1):#最后会取剩下的
            inputs, labels = data
            # wrap them in Variable  
            if use_gpu:  
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  
            else:  
                inputs, labels = Variable(inputs), Variable(labels)             
            
if __name__ == '__main__':
    main()