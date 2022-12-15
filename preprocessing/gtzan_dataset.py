import torch
from torch.utils.data import Dataset
from .stft import load_csv_and_convert
import numpy as np
import math

num_classes = 10
class_dict = {
  'blues': 0,
  'classical': 1,
  'country': 2,
  'disco': 3,
  'hiphop': 4,
  'jazz': 5,
  'metal': 6,
  'pop': 7,
  'reggae': 8,
  'rock': 9,
}

class GTZAN_Dataset(Dataset):
  def __init__(self, load_to_memory=True, mel=False):
    self.load_to_memory = load_to_memory
    self.mel = mel
    tensor_paths, Y = load_csv_and_convert(mel=mel)
    if self.load_to_memory:
      self.images = []
      for tensor_path in tensor_paths:
        self.images.append(self.transform_size(torch.load(tensor_path)))
    self.image_paths = tensor_paths
    self.labels = np.zeros((len(tensor_paths), num_classes))
    for i, _class in enumerate(Y):
      self.labels[i, class_dict[_class]] = 1

  def transform_size(self, image, factor=2):
    if factor == 2:
      return image[:,:2**math.floor(math.log2(image.size()[1])), :2**math.floor(math.log2(image.size()[2]))]
    elif factor == 32:
      return image[:,:int(image.size()[1]/32)*32, :int(image.size()[2]/32)*32]

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    if self.load_to_memory:
      image = self.images[idx]
    else:
      if self.mel:
        image = self.transform_size(torch.load(self.image_paths[idx]), 32)
      else:
        image = self.transform_size(torch.load(self.image_paths[idx]))
    label = self.labels[idx]
    return image, label

# class GTZAN_Prerendered_Dataset(Dataset):
#   def __init__(self, load_to_memory=True):
#     self.load_to_memory = load_to_memory
#     tensor_paths, Y = load_csv_and_convert()
#     if self.load_to_memory:
#       self.images = []
#       for tensor_path in tensor_paths:
#         self.images.append(self.transform_size(torch.load(tensor_path)))
#     self.image_paths = tensor_paths
#     self.labels = np.zeros((len(tensor_paths), num_classes))
#     for i, _class in enumerate(Y):
#       self.labels[i, class_dict[_class]] = 1

#   def transform_size(self, image):
#     return image[:,:2**math.floor(math.log2(image.size()[1])), :2**math.floor(math.log2(image.size()[2]))]

#   def __len__(self):
#     return len(self.image_paths)

#   def __getitem__(self, idx):
#     if self.load_to_memory:
#       image = self.images[idx]
#     else:
#       image = self.transform_size(torch.load(self.image_paths[idx]))
#     label = self.labels[idx]
#     return image, label