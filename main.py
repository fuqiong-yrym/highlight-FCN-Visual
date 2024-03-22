import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
#from typing import Sequence
from torchvision.transforms import functional as F
#import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
#import torchmetrics as TM
from dataclasses import dataclass
import dataclasses
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/content/tensorboard/highlight-detection/')
working_dir = '/content/'
images_folder_name = "test-masks-vis"

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
working_dir = "/content/"

from utils.model import get_fcn_model, get_unet_model
from utils.helpers import *
from utils.dataset import *
from utils.validation import IoULoss
from utils.test import *
from utils.train_epoch import train_model

mode = 'train'
dataset_path = '/content/train_set/'

image_lst, mask_lst = prepareList(dataset_path, mode)


mode = 'test'
dataset_path ='/content/test_set/'
test_image_lst, test_mask_lst = prepareList(dataset_path, mode)

my_test_dataset = SegmentationDataSet(test_image_lst, test_mask_lst)
test_loader = torch.utils.data.DataLoader(my_test_dataset, batch_size = 21)
(test_inputs, test_targets) = next(iter(test_loader))

model = get_fcn_model(pretrained = True, orig = False)

to_device(model)
my_dataset = SegmentationDataSet(image_lst, mask_lst)
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=16, shuffle=True)
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
optimizer = torch.optim.Adam(params, lr = 0.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)

def train_loop(model, train_loader, epochs, optimizer, scheduler, save_path):
    
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, train_loader, test_loader, optimizer, epoch, writer)
        if scheduler is not None:
            scheduler.step()
        

if __name__ == '__main__':
  train_loop(model, train_loader, (1, 5), optimizer, scheduler, save_path=None)
  
  save_path = os.path.join(working_dir, images_folder_name)
  os.makedirs(save_path, exist_ok=True)
  print_test_dataset_masks(model, test_inputs, test_targets, save_path=save_path, show_plot=True)
  torch.save(model.state_dict(), '/content/model.pth')
