# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:29:54 2018

@author: Yun
"""

import os
import shutil

train_rate=0.8

root='D:/Python/pytorch/PyTorchWorkplace/adv_attack_on_transfernet/data/flower_photos'
train_file=os.path.join(root,'train')
test_file=os.path.join(root,'test')

if not os.path.exists(train_file):
    os.makedirs(train_file)
if not os.path.exists(test_file):
    os.makedirs(test_file)

category=['daisy','dandelion','roses','sunflowers','tulips']
for i in range(len(category)):
    child_file=os.path.join(root,category[i])
    child_file_list=[d for d in os.listdir(child_file)]
    for j in range(len(child_file_list)):
#        pic_path=os.path.join(child_file,)
        pic_path=os.path.join(child_file,child_file_list[j])
        if j<len(child_file_list)*train_rate:
            obj_path=os.path.join(train_file,category[i])
        else:
            obj_path=os.path.join(test_file,category[i])
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)    
        shutil.copy(pic_path,obj_path)