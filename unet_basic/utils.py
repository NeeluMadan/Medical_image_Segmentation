import os
import numpy as np
import torch

from skimage import io
from scipy import ndimage

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

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
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample, n_class=5):
        if len(sample['image'].shape) == 2:
            img = np.expand_dims(sample['image'], axis=0)
            if self.labeled:
                mask = np.expand_dims(sample['label'], axis=0)
        elif len(sample['image'].shape) == 3:
            img = np.expand_dims(sample['image'], axis=0)
            if self.labeled:
                mask = np.expand_dims(sample['label'], axis=0)
        else:
            print("Unsupported shape!")

        img = torch.from_numpy(img)

        if self.labeled:
            mask = torch.from_numpy(mask)
            mask = mask.type(torch.LongTensor)
            mask = to_one_hot(mask, n_class)

        if self.labeled:
            return {'image': img, 'label':mask}
        else:
            return {'image': img}

class Rescale:
    def __init__(self, output_size, labeled=True):
        self.output_size = output_size
        self.labeled = labeled

    def __call__(self, sample):
        img = ndimage.interpolation.zoom(sample['image'], zoom=self.output_size, order=1, mode='constant')

        if self.labeled:
            mask = ndimage.interpolation.zoom(sample['label'], zoom=self.output_size, order=0, mode='nearest')

        if self.labeled:
            return {'image': img, 'label':mask}
        else:
            return {'image': img}

class Normalize:
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        if self.labeled:
            mask = sample['label']

        image = sample['image'].astype(np.float32)
        image = 2.*(image - np.min(image))/np.ptp(image)-1

        if self.labeled:
            return {'image': image, 'label':mask}
        else:
            return {'image': image}

class Clahe:
    def __init__(self, clip_limit=0.1, kernel_size=(8, 8)):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)

        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def __call__(self, sample):
        mask = sample['label']
        image = sample['image'].astype(np.float32)
        if not isinstance(image, np.ndarray):
            raise TypeError("Input sample must be a numpy array.")
        input_sample = np.copy(image)
        array = skimage.exposure.equalize_adapthist(
            input_sample,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit
        )

        array = array.astype(np.float32)
        return {'image': array, 'label': mask}

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=(288, 288), p_flip=0.5, p_random_affine=0):
        self.crop = crop
        self.p_flip = p_flip
        self.p_random_affine = p_random_affine

    def __call__(self, sample):
        # transforming to PIL image
        image, mask = F.to_pil_image(sample['image']), F.to_pil_image(sample['label'])

        # random crop
        if self.crop:
            image = F.center_crop(image, self.crop)
            mask = F.center_crop(mask, self.crop)

        # random flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)


        mask = np.array(mask, np.uint8)
        image = np.array(image, np.float32)

        '''
        fig=plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(mask)
        fig.add_subplot(1,2,2)
        plt.imshow(image)
        plt.show()
        plt.close()
        '''

        sample = {'image': image, 'label': mask}
#        sample = ToTensor2(sample)

        return sample
