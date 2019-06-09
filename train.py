import os
import numpy as np
import math
import torchvision
from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
from model import * 
from dataloader import * 


'''
EnhanceNet Implementation by Erik Quintanilla 

Single Image Super Resolution 

https://arxiv.org/abs/1612.07919/
'''

#hyperparams 
cuda = torch.cuda.is_available()
#torch.cuda.empty_cache()
height = 64
width = 64
channels = 3
lr = .0002 
b1 = .5 
b2 = .9
batch_size = 3
n_epochs= 5
hr_shape = (height, width)

#build models 
generator = Generator(residual_blocks=15)
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
	
		#Variables so torch plays nice with autograd, valid and fake for discriminator 
		lr = Variable(lr.type(Tensor))
		hr = Variable(hr.type(Tensor))
		valid = Variable(Tensor(np.ones((batch_size, *discriminator.output_shape))), requires_grad=False)
		fake = Variable(Tensor(np.zeros((batch_size, *discriminator.output_shape))), requires_grad=False)
		
		'''Generator'''
		
		#reset grads 
		g_opti.zero_grad() 
		
		#get our "fake" images from our generator
		generated_hr = generator(lr) 
		
		#what does the discriminator think of these fake images?
		verdict = discriminator(generated_hr) 
		
		#fetch loss
		g_loss = loss(verdict, valid) + loss(generated_hr, hr)
		
		#backpop that loss
		g_loss.backward()
		#update our optimizer 
		g_opti.step()

		'''Discriminator'''

		d_opti.zero_grad() 
		
		#we do a shuffle on the true and fake images so it's not trivial which is which.
		hr_imgs = torch.cat([discriminator(hr), discriminator(generated_hr.detach())], dim=0)
		hr_labels = torch.cat([valid, fake], dim=0)
		idxs = list(range(len(hr_labels)))
		idxs = np.random.shuffle(idxs)
		hr_imgs = hr_imgs[idxs] 
		hr_labels = hr_labels[idxs]

		d_loss = loss(hr_imgs, hr_labels)		
		d_loss.backward() 
		d_opti.step()
		
		print("D: %f G: %f \t Epoch: (%i/%i) Batch: (%i/%i)" %(d_loss.item(), g_loss.item(), epoch, n_epochs, i, len(train_loader)))
		if i % 50 == 0:
			#put the channels back in order!
			generated_hr = generated_hr[:, [2,1,0]]
			#fancy grid so we can view
			generated_hr = make_grid(generated_hr, nrow=1, normalize=True)
			save_image(generated_hr, "samples/%d.png" % i, normalize=False)
		


































