# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:48:32 2019

@author: qzhang40
"""
import os
import random
#from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class SRS_Dataset(Dataset):
	def __init__(self, X,Y = None):
		self.X = X.astype(np.float32)
		self.Y = None        
		if Y is not None:
		    self.Y = Y.astype(np.float32)

	def __getitem__(self, index):
        #get the img and its mask
		x = self.X[index,:,:,:]  
		y = None       
		if self.Y is not None:
		    y = self.Y[index,:,:,:]           
        
#		#x = np.expand_dims(x, axis=0)         
#		x = x.transpose(2,0,1)
#		y = y.transpose(2,0,1)        
#
#		x = torch.from_numpy(x)
#		y = torch.from_numpy(y)
        
        #transforms convert to tensor and to (0,1)
		Transform = []
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
#		
		x = Transform(x)
		if self.Y is not None:  
		    y = Transform(y)            
#        convert img to (-1,1)
#		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#		x = Norm_(x)

		return x, y

	def __len__(self):
		"""Returns the total number of font files."""
		return self.X.shape[0]

def get_loader(X_train, Y_train, batch_size, shuffle = True):
	"""Builds and returns Dataloader."""
	
	dataset = SRS_Dataset(X_train, Y_train)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size, shuffle = shuffle)
	return data_loader





