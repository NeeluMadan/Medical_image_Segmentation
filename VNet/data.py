import numpy as np
import cv2
import os
import torch
import random
import os.path as osp
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase, vol_size):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.vol_size = vol_size
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
            self.masks.append(label_volume_array)
            
            raw_volume_array = sitk.GetArrayFromImage(raw_itk)
            self.images.append(raw_volume_array)
                

    def __len__(self):
        return len(self.images)

        
    def __getitem__(self,item):

        image, label = self.images[item], self.masks[item]
        
        size = np.array(image.shape)

        # Resize whole volume to [128 128 128]
        factor = np.divide(self.vol_size, image.shape)     
        image, label = Rescale(image, label, factor)
        
        # Normalize image w.r.t intensity
        image = Normalize(image)
        
        # Apply elastic tranformation
        if random.choice([True, False]):
            image, label = ElasticDeformation3D(image, label, image.shape[0] * 3,image.shape[0] * 0.05)     
        
        # Apply Random rotation
        if random.choice([True, False]):
            image, label = RandomRotate3D(image, label, (10, 5, 5))
        
        # Apply random flipping 
        if random.choice([True, False]):
            image, label = RandomFlip3D(image, label)
         
        print("shape of image {} and label {}".format(image.shape, label.shape))
        
        image = image[np.newaxis, :, :, :]
        label = label[np.newaxis, :, :, :]
        
        return image, label, size
        #return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(label.astype(np.uint8).copy())

    
    def _check_exists(self):
        return osp.exists(osp.join(self.datapath, "train")) and osp.exists(osp.join(self.datapath, "test"))
    


def Rescale(img, gt, zoom_factor):
    zoom_factor = [0.25, 0.25, 0.25]
    image = ndimage.interpolation.zoom(img, zoom_factor, order=2, mode='constant')   #Resize image  for faster training
    mask = ndimage.interpolation.zoom(gt, zoom_factor, order=0, mode='constant')   #Resize image  for faster training
    
    return image, mask


def Normalize(img):
    image = img.astype(np.float32)
    image = 2.*(image - np.min(image))/np.ptp(image)-1
        
    return image


def ElasticDeformation3D(image, mask, alpha, sigma):
    
    # larger alpha leads to more deformation
    # sigma control the kernel size i.e. larger sigma leads to larger affect area
    """ 
        Elastic deformation of images as described in [Simard2003]_.
        
        
     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
    """
    shape = image.shape
    
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    
    image = map_coordinates(image, indices, order=2).reshape(shape)
    mask = map_coordinates(mask, indices, order=0).reshape(shape)
    
    return image, mask


def RandomRotate3D(image, mask, max_angles = (10, 5, 5)):
    image1 = image
    mask1 = mask
    
    # rotate along z-axis
    angle = random.uniform(-max_angles[0], max_angles[0])
    image2 = rotate(image1, angle, order=2, mode='nearest', axes=(0,1), reshape=False)
    mask2 = rotate(mask1, angle, order=0, mode='nearest', axes=(0,1), reshape=False)
    
    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    image3 = rotate(image2, angle, order=2, mode='nearest', axes=(0,2), reshape=False)
    mask3 = rotate(mask2, angle, order=0, mode='nearest', axes=(0,2), reshape=False)
    
    # rotate along x-axis
    angle = random.uniform(-max_angles[2], max_angles[2])
    rotated_image = rotate(image3, angle, order=2, mode='nearest', axes=(1,2), reshape=False)
    rotated_mask = rotate(mask3, angle, order=0, mode='nearest', axes=(1,2), reshape=False)
    
    return rotated_image, rotated_mask


def RandomFlip3D(image, mask):
    
    if random.choice([True, False]):
        image = image[::-1, :, :].copy()
        mask = mask[::-1, :, :].copy()
        
    if random.choice([True, False]):
        image = image[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()
    
    if random.choice([True, False]):
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()
        
    return image, mask


if __name__ == "__main__":
    segthor_dataset = SegThorDataset(datapath="data",
                                     phase='train',
                                     vol_size=[128, 128, 128])
    
    #print('train_data shape:', segthor_dataset[7])

    for i in range(len(segthor_dataset)):
        image, label, size = segthor_dataset[i]           # Size of image later helps to reshape the test volumes to 
        print(i, image.shape, label.shape)                # their original values
        if i == 5:
            break
