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

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase, transform=None, target_transform=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.transform = transform
        self.target_transform = target_transform
        folder = datapath
        self.images = []
        self.masks = []
        
        if not self._check_exists():
            raise RuntimeError("dataset not found")
                
        if self.phase == 'train':
            folder = datapath + '/train/'
        elif self.phase == 'val':
            folder = datapath + '/val/'
        elif self.phase == 'test':
            folder = datapath + '/test/'         
        
          
            
        for patient in tqdm(os.listdir(folder)):
                
            raw_img = os.path.join(folder, patient, patient+'.nii.gz')   # Reading nifti image
            raw_itk = sitk.ReadImage(raw_img)

            label_img = os.path.join(folder, patient, 'GT.nii.gz')       # Reading Ground Truth labels
            label_itk = sitk.ReadImage(label_img)

            # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
            label_volume_array = sitk.GetArrayFromImage(label_itk)                
            raw_volume_array = sitk.GetArrayFromImage(raw_itk)

            for s in range(1,raw_volume_array.shape[0]):
                # Appending nifti images into an array
                raw_slice_array = raw_volume_array[s-1,:,:]
                self.images.append(raw_slice_array)

                # Appending ground truth labels into an array
                label_slice_array = label_volume_array[s-1,:,:]                    
                self.masks.append(label_slice_array)
                

    def __len__(self):
        return len(self.images)

        
    def __getitem__(self,item):

        image, labels = self.images[item], self.masks[item]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            labels = self.target_transform(labels)

            return image, labels
    
    def _check_exists(self):
        return osp.exists(osp.join(self.datapath, "train")) and osp.exists(osp.join(self.datapath, "test"))
    


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return ndimage.interpolation.zoom(image, zoom=self.output_size)   #Resize image  for faster training
    
    
class Normalize:
    def __call__(self, image):
        image = image.astype(np.float32)     
        lungwin = np.array([-300., 215.])
        image = (image - lungwin[0]) / (lungwin[1] - lungwin[0])
        # intensity normalization [-1,1]
        image = 2.*(image - np.min(image))/np.ptp(image)-1
        # intensity distributed over maximum contrast 
        #image = (ni_max-ni_min)/(i_max-i_min)*(image-i_max)+ni_max
        
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
    segthor_dataset = SegThorDataset(datapath="data",
                                     phase='train',
                                     transform=transforms.Compose([Rescale(0.25)]),
                                     target_transform=transforms.Compose([Rescale(0.25)]))
    
    #print('train_data shape:', segthor_dataset[7])

    for i in range(len(segthor_dataset)):
        image, label = segthor_dataset[i]

        print(i, image.shape, label.shape)
        if i == 5:
            break
