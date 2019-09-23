import cv2
import os
import torch
import skimage
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from utils import JointTransform2D, Rescale, ToTensor, Normalize

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase, patient=None, transform=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.transform = transform
        self.patient = patient
        folder = datapath
        self.images = []
        self.masks = []

        print("phase = ", self.phase)
        if self.phase == 'test':
            raw_img = os.path.join(folder, self.patient)   # Reading nifti image
            raw_itk = sitk.ReadImage(raw_img)
            raw_volume_array = sitk.GetArrayFromImage(raw_itk)
            raw_volume_array = truncated_range(raw_volume_array)

            for s in range(0, raw_volume_array.shape[0]):
                raw_slice_array = raw_volume_array[s,:,:]
                self.images.append(raw_slice_array)

        else: 
            for patient in tqdm(os.listdir(folder)):
                raw_img = os.path.join(folder, patient, patient+'.nii.gz')   # Reading nifti image
                raw_itk = sitk.ReadImage(raw_img)
                raw_volume_array = sitk.GetArrayFromImage(raw_itk)
                raw_volume_array = truncated_range(raw_volume_array)
    
                label_img = os.path.join(folder, patient, 'GT.nii.gz')       # Reading Ground Truth labels
                label_itk = sitk.ReadImage(label_img)
                label_volume_array = sitk.GetArrayFromImage(label_itk)                
                
                # Appending input and GTi images into an array
                for s in range(0, raw_volume_array.shape[0]):
                    raw_slice_array = raw_volume_array[s,:,:]
                    self.images.append(raw_slice_array)
    
                    label_slice_array = label_volume_array[s,:,:]                    
                    self.masks.append(label_slice_array)
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self,item):

        if self.phase == 'test':
            image = self.images[item]
            sample = {'image': image}
    
            torch.manual_seed(1)
            if self.transform:
                sample = self.transform(sample)
    
            return sample

        else:
            image, labels = self.images[item], self.masks[item]
            sample = {'image': image, 'label': labels}

            torch.manual_seed(1)
            if self.transform:
                sample = self.transform(sample)
    
            return sample

def truncated_range(img):
    max_hu = 128
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.

if __name__ == "__main__":

    '''
    ## Loading data for testing phase
    segthor_dataset = SegThorDataset(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data/test",
                                     patient='Patient_58.nii.gz',
                                     phase='test',
                                     transform=transforms.Compose([
                                         Rescale(1.0, labeled=False),
                                         Normalize(labeled=False),
    #                                     ToTensor(labeled=False)
                                    ]))
    
    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]
        
    #    print(i, sample['image'].size())
        plt.imshow(sample['image'])
        plt.show()
        if i == 50:
            break

    '''
#    '''
    ## Loading data for training phase
    segthor_dataset = SegThorDataset(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data_sub/train",
                                     phase='train',
                                     transform=transforms.Compose([
                                         Rescale(1.0, labeled=True),
                                         Normalize(labeled=True),
                                         JointTransform2D(crop=(288, 288), p_flip=0.5),
                                         ToTensor(labeled=True)
                                    ]))
    
    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]
        
        print(i, sample['image'].size(), sample['label'].size())
        if i == 5:
            break
#    '''
