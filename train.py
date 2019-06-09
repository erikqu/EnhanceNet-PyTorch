from model import * 
from dataloader import * 

import os
import numpy as np
import math
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
height = 64
width = 64
channels = 3
lr = .008
b1 = .5 
b2 = .999
batch_size = 3
n_epochs= 5

hr_shape = (height, width)

generator = Generator()
discriminator = Discriminator(input_shape=(channels, *hr_shape))

generator = generator.cuda()
discriminator = discriminator.cuda()
loss = torch.nn.MSELoss().cuda() 

