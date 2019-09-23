import os
import torch
import adabound
from pylab import *
import numpy as np
import torch.nn as nn
from model import UNet
import SimpleITK as sitk
from numpy import ndarray
from scipy import ndimage
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from data import SegThorDataset
from utils import JointTransform2D, Rescale, ToTensor, Normalize

logdir = os.makedirs("models/logs", exist_ok=True)
writer = SummaryWriter(log_dir = logdir)

def tensor_to_numpy(tensor):
    t_numpy = tensor.detach().cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy


# Trying to implement He weight initialization for function ReLu
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
            
            
def dice_loss(result, target, total_classes = 5):
    
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    
    """

    '''
        for j in range(0, total_classes):
            dice_loss_per_class = 1 - dice
            total_loss += dice_loss_per_class

            loss_label[j] = dice_loss_per_class/result.size(0)

    total_loss /= total_classes
    total_loss /= result.size(0)

    '''
    epsilon = 1e-6
    total_loss = 0.0     
    dice_per_class = 0.0
    loss_label =  np.zeros(5)
    weight = [0.2, 2, 0.4, 0.9, 0.8]

    for i in range(result.size(0)):
        Loss = []

        for j in range(0, total_classes):
            result_square_sum = torch.sum(result[i, j, :, :])
            target_square_sum = torch.sum((target[i, j, :, :]).float())
            intersect = torch.sum(result[i, j, :, :] * (target[i, j, :, :]).float())
            dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + intersect + epsilon)
            dice_per_class = 1 - dice
            total_loss += dice_per_class/total_classes
            loss_label[j] += dice_per_class


    loss_label = np.true_divide(loss_label, result.size(0))
        
    return loss_label, total_loss/result.size(0) 

def dice_loss2(result, target, total_classes = 5):
    
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    
    """
    epsilon = 1e-6
    loss_label =  np.zeros(total_classes)
    total_loss = 0
    for j in range(0, total_classes):
            result_square_sum = torch.sum(result[:, j, :, :])
            target_square_sum = torch.sum((target[:, j, :, :]).float())
            intersect = torch.sum(result[:, j, :, :] * (target[:, j, :, :]).float())
            dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + intersect + epsilon)
            dice_loss_per_class = 1 - dice
            total_loss += dice_loss_per_class
            
            loss_label[j] = dice_loss_per_class
            
            
    total_loss /= total_classes
    #total_loss /= result.size(0)
        
    return loss_label, total_loss

def dice_loss3(result, target, batch_size, total_classes = 5):

    epsilon = 1e-6
    target = target.view(batch_size, total_classes, -1).float()
    result = result.view(batch_size, total_classes, -1)

    numerator = 2 * torch.sum(result * target, 2)
    denominator = torch.sum(result + target**2, 2) + epsilon

    return 1 - torch.mean(numerator / denominator)


def train(epochs, batch_size, learning_rate):

    torch.manual_seed(1234)

    train_loader = torch.utils.data.DataLoader(
        SegThorDataset("/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data/train", phase='train',
                       transform=transforms.Compose([
                           Rescale(1.0),
                           Normalize(),                           
                           ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    '''
    # Loading validation data
    val_set = SegThorDataset("/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data_val", phase='val',
                                   transform=transforms.Compose([
                                       Rescale(0.5),
                                       Normalize(),
                                       ToTensor2()
                                   ]))

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=1,
                                             shuffle=False)
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.apply(weight_init)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #learning rate to 0.001 for initial stage
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.00001)
    #optimizer = adabound.AdaBound(params = model.parameters(), lr = 0.001, final_lr = 0.1)
    
    for epoch in range(epochs):
        f = open('train_output.log', 'a')
        f.write('Epoch {}/{}\n'.format(epoch + 1, epochs))
        f.write('-' * 10)
        
        running_loss = 0.0
        running_loss_label = np.zeros(5) 
        for batch_idx, sample in enumerate(train_loader):
            train_data, labels = sample['image'].to(device, dtype=torch.float), sample['label'].to(device, dtype=torch.uint8)

            optimizer.zero_grad()         
            output = model(train_data)       
            
            loss_label, loss = dice_loss2(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for i in range(5):
                running_loss_label[i] += loss_label[i]

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        f.write("\n Total Dice Loss: {:.4f}\n".format(epoch_loss))
        epoch_loss_class = np.true_divide(running_loss_label, len(train_loader))
        f.write("Dice per class: Background = {:.4f} Eusophagus = {:.4f}  Heart = {:.4f}  Trachea = {:.4f}  Aorta = {:.4f}\n".format(epoch_loss_class[0], epoch_loss_class[1], epoch_loss_class[2], epoch_loss_class[3], epoch_loss_class[4]))
        #f.write("Dice per class: Background = {:.4f} Eusophagus = {:.4f}\n".format(epoch_loss_class[0], epoch_loss_class[1]))
        f.close()

        if epoch%4==0:
            os.makedirs("models", exist_ok=True)
            torch.save(model, "models/model.pt")

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/model.pt")


def evaluate_model(model, val_loader, val_set, epoch, device):

    model = torch.load("models/model.pt")
    model.eval()

    count = 0
    seg_vol = zeros([len(val_set),  512, 512])
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            val_data, val_labels = sample['image'].to(device, dtype=torch.float), sample['label'].to(device, dtype=torch.uint8)

            output = model(val_data)

            max_idx = torch.argmax(output, 1, keepdim=True)
            max_idx = tensor_to_numpy(max_idx)

            slice_v = max_idx[:,:]
            slice_v = slice_v.astype(float32)
            slice_v = ndimage.interpolation.zoom(slice_v, zoom=2, order=0, mode='nearest', prefilter=True)
            seg_vol[count,:,:] = slice_v
            count = count + 1

        os.makedirs("validation_result", exist_ok=True)
        filename = os.path.join('validation_result', 'Patient_11_'+str(epoch)+'.nii')
        segmentation = sitk.GetImageFromArray(seg_vol, isVector=False)
        print("Saving segmented volume of size: ",segmentation.GetSize())
        sitk.WriteImage(sitk.Cast( segmentation, sitk.sitkUInt8 ), filename, True)


if __name__ == "__main__":
    train(epochs=50, batch_size=4, learning_rate=0.01)
