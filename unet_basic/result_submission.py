import numpy as np
import cv2
import os
import torch
import os.path as osp
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms

class SegThorSubmission(Dataset):
    def __init__(self, datapath, patient, phase, transform=None, target_transform=None):
        self.phase = phase
        self.datapath = datapath
        self.patient = patient
        self.transform = transform
        self.target_transform = target_transform
        folder = datapath
        self.images = []
      
                
        raw_img = os.path.join(datapath, patient)   # Reading nifti image
        print(raw_img)
        raw_itk = sitk.ReadImage(raw_img)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x             
        raw_volume_array = sitk.GetArrayFromImage(raw_itk)
        print("shape of the volume while reading ", raw_volume_array.shape)
        
        for s in range(0,raw_volume_array.shape[0]):
            # Appending nifti images into an array
            raw_slice_array = raw_volume_array[s,:,:]
            self.images.append(raw_slice_array)

        print("Length after assinging to list", len(self.images))
    def __len__(self):
        return len(self.images)

        
    def __getitem__(self,item):

        image= self.images[item]

        if self.transform:
            image = self.transform(image)

            return image


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return ndimage.interpolation.zoom(image, zoom=self.output_size)   #Resize image  for faster training
    
    
class Normalize:
    def __call__(self, image):
        image = image.astype(np.float32)
        win = np.array([-700., 225.])
        image = (image - win[0]) / (win[1] - win[0])
        image[image < 0] = 0
        image[image > 1] = 1

        return image


class ToTensor:
    def __call__(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            data = data.transpose((2, 0, 1))
        else:
            print("Unsupported shape!")
        return torch.from_numpy(data)
    
    
if __name__ == "__main__":
    patient = "Patient_58.nii"
    segthor_dataset = SegThorSubmission(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data", 
                                     patient='Patient_58.nii',
                                     phase='test',
                                     transform=transforms.Compose([Rescale(0.25)]),
                                     target_transform=transforms.Compose([Rescale(0.25)]))
    
    print('Length of dataset:', len(segthor_dataset))

    for i in range(len(segthor_dataset)):
        image = segthor_dataset[i]

        print(i, image.shape)
        if i == 5:
            break
