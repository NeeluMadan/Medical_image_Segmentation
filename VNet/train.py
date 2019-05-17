#!/usr/bin/env python3

import os
import torch
import adabound
from pylab import *
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import SimpleITK as sitk
from numpy import ndarray
from copy import deepcopy
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from model import VNet
from data import SegThorDataset




def dice_loss(result, target, total_classes = 5):
    
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    
    """
    epsilon = 1e-6
    loss = 0.0     
    loss_label =  np.zeros(5)
    weight = [0.2, 2, 0.4, 0.9, 0.8]
    
    for i in range(result.size(0)):
        Loss = []

        for j in range(0, total_classes):
            result_square_sum = torch.sum(result[i, j, :, :, :])
            target_square_sum = torch.sum((target[i, :, :, :, :] == j))
            intersect = torch.sum(result[i, j, :, :, :] * (target[i, :, :, :, :] == j).float())
            dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + intersect + epsilon)
            Loss.append(1 - dice)
        
        for i in range(5):
            loss += Loss[i] * weight[i]        
            loss_label[i] += Loss[i]

    loss_label = np.true_divide(loss_label, result.size(0))

    return loss_label, loss/result.size(0)



def train(epochs, batch_size, learning_rate):
    
    train_set = SegThorDataset("data", phase = 'train', vol_size = [128, 128, 128])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet().to(device)
    
   # print(model)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #learning rate to 0.001 for initial stage
    #optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.95)
    optimizer = adabound.AdaBound(params = model.parameters(), lr = 0.001, final_lr = 0.1)
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        
        running_loss = 0.0
        loss_seg =  np.zeros(5)
        
        for batch_idx, (train_data, labels, size) in enumerate(train_loader):
            train_data, labels = train_data.to(device, dtype=torch.float), labels.to(device, dtype=torch.uint8)

            optimizer.zero_grad()         
            output = model(train_data)       
            
            loss_label, loss = dice_loss(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            for i in range(4):
                loss_seg[i] += loss_label[i]


        epoch_loss = running_loss / len(train_loader)
        epoch_loss_class = np.true_divide(loss_seg, len(train_loader)) 
        print("Dice per class: Background = {:.4f} Eusophagus = {:.4f}  Heart = {:.4f}  Trachea = {:.4f}  Aorta = {:.4f}\n".format(epoch_loss_class[0], epoch_loss_class[1], epoch_loss_class[2], epoch_loss_class[3], epoch_loss_class[4]))
        print("Total Dice Loss: {:.4f}\n".format(epoch_loss))

    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/model.pt")
    
    
if __name__ == "__main__":
    train(epochs=200, batch_size=2, learning_rate=0.001)
