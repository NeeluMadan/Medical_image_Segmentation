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
    def __init__(self, datapath, phase, transform=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.transform = transform
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

            for s in range(0,raw_volume_array.shape[0]):
                # Appending nifti images into an array
                raw_slice_array = raw_volume_array[s,:,:]
                self.images.append(raw_slice_array)

                # Appending ground truth labels into an array
                label_slice_array = label_volume_array[s,:,:]                    
                self.masks.append(label_slice_array)
                

    def __len__(self):
        return len(self.images)

        
    def __getitem__(self,item):

        image, labels = self.images[item], self.masks[item]

        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _check_exists(self):
        return osp.exists(osp.join(self.datapath, "train")) and osp.exists(osp.join(self.datapath, "test"))


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
#    print("print before applying one hot: ", y_one_hot.size())
    y_one_hot = y_one_hot.scatter(0, mask, 1).long()
    return y_one_hot


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
        lungwin = np.array([-128., 384.])
        #lungwin = np.array([-1200., 600.])
        image = (image - lungwin[0]) / (lungwin[1] - lungwin[0])
        image = 2.*(image - np.min(image))/np.ptp(image)-1
        image[image < 0] = 0
        image[image > 1] = 1

        return {'image': image, 'label': mask}


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
    segthor_dataset = SegThorDataset(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data_sub",
                                     phase='train',
                                     transform=transforms.Compose([Rescale(0.5),
                                         Normalize(),
                                         ToTensor2()
                                         ]))
    
    #print('train_data shape:', segthor_dataset[7])

    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]

        print(i, sample['image'].size(), sample['label'].size())
        if i == 5:
            break

