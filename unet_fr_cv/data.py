import numpy as np
import cv2
import os
import torch
import ntpath
import os.path as osp
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from skimage import exposure
from torch.utils.data import Dataset
from torchvision import transforms

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase, transform=None, file_list=None):

        self.phase = phase
        self.datapath = datapath
        self.datapath_list = []
        self.transform = transform
        self.file_list = file_list
        folder = datapath
        num_slice = []
        patient_list = []
        idx_list = []

        if not self._check_exists():
            raise RuntimeError("dataset not found")

        for f in tqdm(os.listdir(datapath)):
            if not f in self.file_list:
                continue

            patient = os.path.join(folder, f)
            patient_list.append(patient)

            raw_img = os.path.join(folder, f, f)
            patient_im = sitk.ReadImage(raw_img)

            num_slice = patient_im.GetDepth()

            for i in range(num_slice):
                self.datapath_list.append( (patient, i, num_slice) )
            

    def __len__(self):
        return len(self.datapath_list)

        
    def __getitem__(self,item):
        cur_patient = self.datapath_list[item][0]
        cur_slice_idx = self.datapath_list[item][1]
        cur_slice_count = self.datapath_list[item][2]
        
        raw_img = os.path.join(cur_patient, ntpath.basename(cur_patient)+'.nii.gz')                         # Reading nifti image
        raw_itk = sitk.ReadImage(raw_img)

        label_img = os.path.join(cur_patient, 'GT.nii.gz')                                                  # Reading Ground Truth labels
        label_itk = sitk.ReadImage(label_img)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        label_volume_array = sitk.GetArrayFromImage(label_itk)
        raw_volume_array = sitk.GetArrayFromImage(raw_itk)
        raw_volume_array = truncated_range(raw_volume_array)

        patient_img = raw_volume_array[cur_slice_idx,:,:]
        patient_gt = label_volume_array[cur_slice_idx,:,:]

        sample = {'image': patient_img, 'label': patient_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
               
    def _check_exists(self):
        return osp.exists(osp.join(self.datapath))


def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img = ndimage.interpolation.zoom(sample['image'], zoom=self.output_size, order=1, mode='constant')
        mask = ndimage.interpolation.zoom(sample['label'], zoom=self.output_size, order=0, mode='nearest')

        return {'image': img, 'label':mask}
    
class Normalize:
    def __call__(self, sample):

        mask = sample['label']
        image = sample['image'].astype(np.float32)     
        image = 2.*(image - np.min(image))/np.ptp(image)-1

        return {'image': image, 'label':mask}

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def to_one_hot(mask, n_class):
    """
    Transform a mask to one hot
    Args:
        mask:
        n_class: number of class for segmentation
    Returns:
        y_one_hot: one hot mask
    """
    y_one_hot = torch.zeros((n_class, mask.shape[1], mask.shape[2]))
    y_one_hot = y_one_hot.scatter(0, mask, 1).long()
    return y_one_hot


class ToTensor:
    def __call__(self, sample):
        if len(sample['image'].shape) == 2:
            img = np.expand_dims(sample['image'], axis=0)
            mask = np.expand_dims(sample['label'], axis=0)
        elif len(sample['image'].shape) == 3:
            img = sample['image'].transpose((2, 0, 1))
            mask = sample['label'].transpose((2, 0, 1))
        else:
            print("Unsupported shape!")

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return {'image': img, 'label':mask}
    

class ToTensor2:
    def __call__(self, sample, n_class=5):
        if len(sample['image'].shape) == 2:
            img = np.expand_dims(sample['image'], axis=0)
            mask = np.expand_dims(sample['label'], axis=0)
        elif len(sample['image'].shape) == 3:
            img = np.expand_dims(sample['image'], axis=0)
            mask = np.expand_dims(sample['label'], axis=0)
        else:
            print("Unsupported shape!")

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        mask = mask.type(torch.LongTensor)
        mask = to_one_hot(mask, n_class)
        return {'image': img, 'label':mask}


if __name__ == "__main__":

    segthor_dataset = SegThorDataset(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data_sub/train",
                                     phase='train',
                                     transform=transforms.Compose([Rescale(1.0),
                                         Normalize(),
                                         ToTensor2()
                                         ]), file_list=['Patient_24'])
    
    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]

        print(i, sample['image'].size(), sample['label'].size())
        if i == 5:
            break

