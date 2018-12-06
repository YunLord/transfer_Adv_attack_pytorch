# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:03:23 2018

@author: YunLo
"""

from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torchvision
from os.path import join

from torch.utils.data import DataLoader

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data,attack_over_test_data_and_save
    
parser = argparse.ArgumentParser(description='cycle_advGan')
parser.add_argument('--target-model', type=str, default='LeNet5',metavar='N',
                    help='the attack model')
parser.add_argument('--pretrained-model', type=str, default='mnist_lenet.pth',metavar='N',
                    help='load pretrained model')
parser.add_argument('--dataset', type=str, default='mnist',metavar='N',
                    help='dataset')
parser.add_argument('--shuffle', type=str, default='Fasle',metavar='N',
                    help='dataloder is shuffle ')
parser.add_argument('--batch-size', type=int, default=100, metavar='N', # batch_size参数，如果想改，如改成128可这么写：python main.py -batch_size=128
                    help='input batch size for training (default: 64)')
parser.add_argument('--use-cuda', action='store_true', default=True, # GPU参数，默认为False
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.3)
parser.add_argument('--num_label', type=int, default=10)
parser.add_argument('--img-save-path', type=str, default='advBIM',metavar='N',
                    help='save the adv img')
parser.add_argument('--attack', type=str, default='FGSM',metavar='N',
                    help='attack method')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--mode', type=str, default='test', help='mode')

args = parser.parse_args()  # 这个是使用argparse模块时的必备行，将参数进行关联，详情用法请百度 argparse 即可

device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

def load_targeted_model():
    if args.target_model == "LeNet5":
        from target_models.lenet5 import LeNet5
        model = LeNet5()
    if args.target_model == "resnet18":
        from target_models.resnet import ResNet18
        model = ResNet18()      
    return model

def make_dataset(mode='train'):
    # Small noise is added, following SN-GAN
#    def noise(x):
#        return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
    istrain=True
    if (mode!='train'):
        istrain=False
    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='datasets/cifar10_data', train=istrain, transform=transforms.ToTensor(), download=True)
        model_num_labels=10
    if args.dataset == "mnist":
        dataset = torchvision.datasets.MNIST('datasets/MNIST_data', train=istrain, transform=transforms.ToTensor(), download=True)
        model_num_labels=10
#    if args.dataset == "mnist_adv":
#        dataset=ImageDataset(args, transforms_=transforms.ToTensor(), unaligned=False,mode=mode)
#        model_num_labels=10
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
#    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

#    return train_dataloader,test_dataloader,model_num_labels
    return dataset,dataloader,model_num_labels

def attack(targeted_model, random_start=False,args):
    if args.attack=='FGSM':
        from adversarialbox.attacks import FGSMAttack
        adversary=FGSMAttack(targeted_model,args.epsilon)
    if args.attack=='BIM':
        from adversarialbox.attacks import LinfPGDAttack
        adversary=LinfPGDAttack(targeted_model, random_start)
        
def load_pretrained_model(model):
    pretrained_model =join('pretrained_model',args.pretrained_model)
    model.load_state_dict(torch.load(pretrained_model))
    return model

if __name__ == "__main__":
    targeted_model=load_targeted_model().to(device)
    targeted_model=load_pretrained_model(targeted_model)
#    for p in targeted_model.parameters():
#        p.requires_grad = False
    targeted_model.eval()
    dataset,dataloader,data_num_labels=make_dataset(args.mode)
    test(targeted_model, dataloader)
    # Adversarial attack
#    adversary = FGSMAttack(targeted_model, args.epsilon)
    adversary = attack(targeted_model, random_start=False,args)
    t0 = time()
#    attack_over_test_data(targeted_model, adversary,train_dataloader ,args)
    attack_over_test_data_and_save(targeted_model, adversary,dataset ,args)
    print('{}s eclipsed.'.format(time()-t0))
