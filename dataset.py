import torch
#from skimage.io import imread
from torchvision.io import read_image
from torch.utils import data
import torchvision.transforms as T
import numpy as np
import numpy.ma as ma

def prepareList(base_dir, mode):
  #base_dir = '/content/train_set/'
  with open('/content/'+ mode +'.lst', "r") as f:
      lines = f.read().splitlines()
  n = len(lines) # 3017 for training
  splits = [lines[i].split() for i in range(n)]
  inputs = [splits[i][0] for i in range(n)]
  targets = [splits[i][1] for i in range(n)]
  img_paths = [base_dir + inputs[i] for i in range(n)]
  mask_paths = [base_dir + targets[i] for i in range(n)]
  return img_paths[:50], mask_paths[:50] # choose first 100 samples for try out

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 #transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 transform = None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = read_image(input_ID), read_image(target_ID)
        if x.shape[0] == 4:
          x = x[:3]
        if y.shape[0] >= 2:
          y = y[0].unsqueeze(0)
        x = x/225. # convert (0,255) to (0,1)
        #Genearte binary mask
        y_np = np.array(y)
        y = torch.from_numpy(ma.make_mask(y_np).astype(int))
        #y = y/225.
        # Preprocessing
        # check how to differciate transforms for x and y
        resize = T.Resize((256,256),interpolation=T.InterpolationMode.NEAREST)
        x = resize(x)
        y = resize(y)
        if self.transform is not None:
            x = self.transform(x)
            #y = self.transform(y) # y does not need normalization
        # Typecasting
        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return x, y
  



