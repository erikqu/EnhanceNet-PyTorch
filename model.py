import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from numba import jit

class GELU(nn.Module):
	"""
	GELU activation approx. courtesy of 
	https://arxiv.org/pdf/1606.08415.pdf
	
	(note numba)
	"""
	@jit
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ResidualBlock(nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
			GELU(),
			nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
		)

	def forward(self, x):
		return self.conv_block(x)
		
		
class Generator(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, residual_blocks=10):
		super(GeneratorResNet, self).__init__()
		self.conv1 = nn.Sequential(
						nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), 
						GELU())

		# Residual blocks
		residuals = []
		for _ in range(residual_blocks):
			residuals.append(ResidualBlock(64))
		self.residuals = nn.Sequential(*residuals)
		
		#nearest neighbor upsample 
		self.upsample = nn.sequential(
				nn.Upsample(scale_factor=2),
				nn.Conv2d(64, 64, 3, 1, 1),
				GELU(),
				nn.Upsample(scale_factor=2),
				nn.Conv2d(64, 64, 3, 1, 1),
				GELU())
		self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4), GELU())
		self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4))

	def forward(self, x):
		out = self.conv1(x)
		out = self.residuals(out1)
		out = self.conv2(out)
		i_bicubic= self.upsample(out)
		out = self.conv3(out)
		i_res = self.conv4(out) 
		
		return out
		
		
		
		
		
		
		