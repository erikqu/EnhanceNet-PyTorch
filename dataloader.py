from torch.utils.data import Dataset
import torch
import cv2
import numpy as np 

class Dataset(Dataset):
	'''
	LR = low resolution image
	HR = high resolution image 
	dir = dump of image directories for dataset
	
	'''
	def __init__(self, ids, lr,hr):
		'Initialization'
		self.dir = ids
		self.lr = lr
		self.hr = hr 
	def __len__(self):
		return len(self.dir)
	def __getitem__(self, index):
		filename = self.dir[index] 
		lower = cv2.imread(self.lr + filename,1)
		higher = cv2.imread(self.hr  + filename,1) 
		#transpose so pytorch plays nice 
		lower= lower.transpose((2, 0, 1))
		higher = higher.transpose((2, 0, 1))		
		#pass numpy arrays to torch and make float tensors.
		lower = torch.from_numpy(lower).float()
		higher =  torch.from_numpy(higher).float()
		return lower, higher