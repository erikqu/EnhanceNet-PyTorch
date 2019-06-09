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
import torch

#hyperparams 
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

#build models 
generator = Generator()
discriminator = Discriminator(input_shape=(channels, *hr_shape))

#send to gpu 
generator = generator.cuda()
discriminator = discriminator.cuda()
loss = torch.nn.MSELoss().cuda() 

#set optimizers 
g_opti = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
d_opti = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

#set data geneator 
curmovie = "socialnetwork"
imagedir = np.load(curmovie + "_ids.npy")
lowres = "M:/Experiments/OLSS/frames/" + curmovie + "_128/"
highres = "M:/Experiments/OLSS/frames/" + curmovie + "_512/"
gen = Dataset(ids = imagedir, lr = lowres, hr = highres)
train_loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = iter(train_loader)

Tensor = torch.cuda.FloatTensor

for epoch in range(n_epochs):
	for i, (lr, hr) in enumerate(train_loader):
		lr = Variable(lr.type(Tensor))
		hr = Variable(hr.type(Tensor))
		valid = Variable(Tensor(np.ones((batch_size, *discriminator.output_shape))), requires_grad=False)
		fake = Variable(Tensor(np.zeros((batch_size, *discriminator.output_shape))), requires_grad=False)
		
		#Generator 
		g_opti.zero_grad() 
		
		generated_hr = generator(lr) 
		
		verdict = discriminator(generated_hr) 
		
		g_loss = loss(verdict, valid) 
		
		g_loss.backward() 
		g_opti.step()

		#Discriminator 

		d_opti.zero_grad() 
		hr_imgs = torch.cat([discriminator(hr), discriminator(generated_hr.detach())], dim=0)
		hr_labels = torch.cat([valid, fake], dim=0)
		
		idxs = list(range(len(hr_labels)))
		idxs = np.random.shuffle(idxs)
		hr_imgs = hr_imgs[idxs] 
		hr_labels = hr_labels[idxs]

		d_loss = loss(hr_imgs, hr_labels)
		
		d_loss.backward() 
		d_opti.step()
		
		print(d_loss)

		


































