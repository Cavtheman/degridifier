import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
import random

from gridifier import *

from PIL import Image

class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, list_IDs, label_path, grid_size, grid_intensity, grid_offset_x, grid_offset_y, hflip=None, vflip=None, angle=0, shear=0, brightness=1, pad=(0,0,0,0), contrast=1, use_channel=None):
        self.list_IDs = list_IDs
        self.label_path = label_path
        self.grid_size = grid_size
        self.grid_intensity = grid_intensity
        self.grid_offset_x = grid_offset_x
        self.grid_offset_y = grid_offset_y
        self.hflip = hflip
        self.vflip = vflip
        self.angle = angle
        self.shear = shear
        self.brightness = brightness
        self.pad = pad
        self.contrast = contrast
        self.use_channel = use_channel

    def __augment_data__(self, labels, grid_size, grid_intensity, grid_offset_x, grid_offset_y, hflip, vflip, angle, shear, brightness, pad, contrast, use_channel):

        grid_size = random.randint(grid_size[0], grid_size[1])
        grid_intensity = random.uniform(grid_intensity[0], grid_intensity[1])
        grid_offset_x = random.randint(grid_offset_x[0], grid_offset_x[1])
        grid_offset_y = random.randint(grid_offset_y[0], grid_offset_y[1])

        data = Image.fromarray(np.uint8(gridify(labels, grid_size, grid_intensity, grid_offset_x, grid_offset_y) * 255))

        # Flipping the image horizontally
        if hflip and random.random() > hflip:
            data = TF.hflip(data)
            labels = TF.hflip(labels)

        # Flipping the image vertically
        if vflip and random.random() > vflip:
            data = TF.vflip(data)
            labels = TF.vflip(labels)

        # Adjusts the brightness randomly
        if type(brightness) is tuple:
            brightness = random.uniform(brightness[0], brightness[1])

        # Rotates image randomly
        if type(angle) is tuple:
            angle = random.randint(angle[0], angle[1])

        # Shears image randomly
        if type(shear) is tuple:
            shear = random.uniform(shear[0], shear[1])

        if type(contrast) is tuple:
            contrast = random.uniform(contrast[0], contrast[1])

        data = TF.adjust_contrast(data, contrast)
        data = TF.adjust_brightness(data, brightness)

        data = TF.affine(data, angle, (0,0), 1, shear)
        labels = TF.affine(labels, angle, (0,0), 1, shear)

        data = F.pad(ToTensor()(data), pad)
        labels = F.pad(ToTensor()(labels), pad)

        if use_channel:
            data = torch.narrow(data, 0, use_channel, 1)

        return data, labels


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        ID = self.list_IDs[index]

        # Load data and get label
        labels = Image.open(self.label_path.format(ID))

        # Augment the data and labels randomly using given arguments
        data, labels = self.__augment_data__(labels, self.grid_size, self.grid_intensity, self.grid_offset_x, self.grid_offset_y, self.hflip, self.vflip, self.angle, self.shear, self.brightness, self.pad, self.contrast, self.use_channel)

        return data, labels
