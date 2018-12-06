# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:48:30 2018

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

def train_model(model, criterion, optimizer,dsets,use_gpu):  
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}  
    dset_classes = dsets['train'].classes      
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=16,  
                                               shuffle=True, num_workers=0)  
                for x in ['train', 'val']}  
    since = time.time()  
      
#        best_model = model  
#    val_acc = 0.0  
  
##        for epoch in range(num_epochs):  
#            print('Epoch {}/{}'.format(epoch, num_epochs - 1))  
#            print('-' * 10)  
  
        # Each epoch has a training and validation phase  
    for phase in ['train', 'val']:  
        if phase == 'train':  
            model.train(True)  # Set model to training mode  
        else:  
            model.train(False)  # Set model to evaluate mode  
  
        running_loss = 0.0  
        running_corrects = 0  
  
        # Iterate over data.  
        for data in dset_loaders[phase]:  
            # get the inputs  
            inputs, labels = data  
  
            # wrap them in Variable  
            if use_gpu:  
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  
            else:  
                inputs, labels = Variable(inputs), Variable(labels)  
  
            # zero the parameter gradients  
            optimizer.zero_grad()  
  
            # forward  
            outputs = model(inputs)  
            _, preds = torch.max(outputs.data, 1)  
            loss = criterion(outputs, labels)  
  
            # backward + optimize only if in training phase  
            if phase == 'train':  
                loss.backward()  
                optimizer.step()  
  
            # statistics  
            running_loss += loss.item()  
            running_corrects += torch.sum(preds == labels.data)  
  
        epoch_loss = running_loss / dset_sizes[phase]  
        epoch_acc = running_corrects / dset_sizes[phase]  
  
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(  
            phase, epoch_loss, epoch_acc))  
  
        # deep copy the model  
        if phase == 'val':  
            val_acc = epoch_acc  
  
    print()  
  
    time_elapsed = time.time() - since  
    print('Training complete in {:.0f}m {:.0f}s'.format(  
        time_elapsed // 60, time_elapsed % 60))  
    print('val Acc: {:4f}'.format(val_acc))  
    return model   
    
def main():
    data_dir = 'catdogdata'  
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  
             for x in ['train', 'val']}  

      
    use_gpu = False        
    model_ft = models.resnet18(pretrained=True)  
    num_ftrs = model_ft.fc.in_features  
    model_ft.fc = nn.Linear(num_ftrs, 2)  
      
    if use_gpu:  
        model_ft = model_ft.cuda()  
      
    criterion = nn.CrossEntropyLoss()  
      
    # Observe that all parameters are being optimized  
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) 
    num_epoch = 20
    for epoch in range(num_epoch):
        print('{}/{}'.format(epoch + 1, num_epoch))
        print('-' * 10)
        model_ft = train_model(model_ft, criterion, optimizer_ft,dsets,use_gpu)
        print('Finish Training!')
        print()
#        save_path = os.path.join(root, 'model_save')
        save_path=('model_save')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model_ft.state_dict(), save_path +'/'+str(epoch)+'.pth')
if __name__ == '__main__':
    main()