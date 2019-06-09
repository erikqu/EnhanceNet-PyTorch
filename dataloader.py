import torch
from torch.utils import data
from torch.utils.data import DataLoader
import cv2

class Dataset(data.Dataset):
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
		file = self.dir[index] 
		lower = cv2.imread(self.lr + file,1)
		higher = cv2.imread(self.hr  + file,1) 
		lower = np.asarray(lower) 
		higher = np.asarray(higher)
		#print(lower.shape)
		lower= lower.transpose((2, 0, 1))
		higher = higher.transpose((2, 0, 1))		
		lower = torch.from_numpy(lower)
		higher =  torch.from_numpy(higher)
		return lower, higher