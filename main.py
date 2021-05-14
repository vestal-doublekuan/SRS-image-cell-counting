import os
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import imageio

import torch
import torch.utils.data as data
from torchvision import transforms as T

#from model.U_net_pytorch import U_Net_2Ch
from data_loader import get_loader
from solver import Solver




train_loader = get_loader(X_train_tra, Y_train_tra, 16, True)
train_val_loader = get_loader(X_train_val, Y_train_val, 16, True)

solver = Solver(train_loader, train_val_loader,
                     path=directory, mode='SRS',
                     num_epochs=100,
                     batch_size=16, shuffle=True,
                     num_epochs_decay=0, save_val_best=False, stop_early=True)

hist = solver.train()
