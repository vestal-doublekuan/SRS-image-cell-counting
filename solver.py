# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:41:15 2019

@author: qzhang40
"""
import os
import numpy as np
from time import time
from datetime import datetime
import csv
import pandas as pd

import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset


from model.U_net_pytorch import U_Net_2ch as U_Net
from model.loss import DiceLoss
from model.evaluation import *
from model.EarlyStopping import EarlyStopping

class Solver(object):
    
    def __init__(self, train_loader = None, valid_loader = None, 
                 path = None, mode = 'SRS', num_epochs = 300, 
                 batch_size = 16, shuffle = True, 
                 num_epochs_decay = 0, save_val_best = True, stop_early = True):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.criterion = torch.nn.BCELoss()

		# Hyper-parameter
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999

		# Training settings
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs_decay = num_epochs_decay
        self.save_val_best = save_val_best
        self.stop_early =stop_early
        
		# Path
        self.result_path = path
        self.mode = mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'U_Net'
        self.build_model(show = False)
        
        
        
    def build_model(self, show = True):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=2,output_ch=1)
        else:
            raise Exception('Mode is not defined!')
			

        self.optimizer = optim.Adam(self.unet.parameters(),
									  self.lr)#, [self.beta1, self.beta2])
        self.unet.to(self.device)
        
        if show:
            self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        


    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()       

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img        
    
    def load_model(self):
        unet_path = os.path.join(self.result_path, '%s_%d_%d_%.4f_%d.pkl' %(self.model_type,self.num_epochs,self.batch_size,self.lr,self.num_epochs_decay))
        self.unet.load_state_dict(torch.load(unet_path))       



    
    def train(self):
        """Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

        unet_path = os.path.join(self.result_path, '%s_%d_%d_%.4f_%d.pkl' %(self.model_type,self.num_epochs,self.batch_size,self.lr,self.num_epochs_decay))
        hist = []
        lr = self.lr
        best_unet_score = 0.   
        
        print ('trainning..')	
        
        early_stopping = EarlyStopping(patience=25)
        
        
        for epoch in range(int(self.num_epochs)):
        
            self.unet.train(True)
            epoch_loss = 0
        				
            acc = 0.	# Accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length = 0   
            for i, (images, GT) in enumerate(self.train_loader):
                # GT : Ground Truth
#                if i >0:
#                    break
                #images = images.to(self.device)
                #GT = GT.to(self.device)
                
                
                images = images.cuda()
                GT = GT.cuda()
            
                # SR : Segmentation Result
                SR = self.unet(images)
#                SR_probs = F.sigmoid(SR)
                SR_flat = SR.view(SR.size(0),-1)
            
                GT_flat = GT.view(GT.size(0),-1)
                loss = self.criterion(SR_flat,GT_flat)
                #print (loss.item())
                epoch_loss += loss.item()
            
                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()
            
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)
                length += 1
        
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            JS = JS/length
            DC = DC/length      
            unet_score = JS + DC
            epoch_loss = epoch_loss/length
                    
            # Print the log info
            print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                    epoch+1, self.num_epochs, \
                    epoch_loss,\
                    acc,SE,SP,PC,F1,JS,DC))        
                    
            
				# Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print ('Decay learning rate to lr: {}.'.format(lr))

    
				#===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()
            
            epoch_loss_v = 0    
            
            acc_v = 0.	# Accuracy
            SE_v = 0.		# Sensitivity (Recall)
            SP_v = 0.		# Specificity
            PC_v = 0. 	# Precision
            F1_v = 0.		# F1 Score
            JS_v = 0.		# Jaccard Similarity
            DC_v = 0.		# Dice Coefficient
            length=0
            
            for i, (images, GT) in enumerate(self.valid_loader):
                
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images)
                
                SR_flat = SR.view(SR.size(0),-1)
                GT_flat = GT.view(GT.size(0),-1)
                loss = self.criterion(SR_flat,GT_flat)
                epoch_loss_v += loss.item()                
                
                
                acc_v += get_accuracy(SR,GT)
                SE_v += get_sensitivity(SR,GT)
                SP_v += get_specificity(SR,GT)
                PC_v += get_precision(SR,GT)
                F1_v += get_F1(SR,GT)
                JS_v += get_JS(SR,GT)
                DC_v += get_DC(SR,GT)
                
                length += 1
                
            acc_v = acc_v/length
            SE_v = SE_v/length
            SP_v = SP_v/length
            PC_v = PC_v/length
            F1_v = F1_v/length
            JS_v = JS_v/length
            DC_v = DC_v/length
            unet_score_v = JS_v + DC_v
            epoch_loss_v = epoch_loss_v/length
            
            print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, Loss: %.4f'%(acc_v,SE_v,SP_v,PC_v,F1_v,JS_v,DC_v, epoch_loss_v))
            hist.append(np.array([epoch+1,acc,SE,SP,PC,F1,JS,DC,unet_score, epoch_loss, acc_v,SE_v,SP_v,PC_v,F1_v,JS_v,DC_v,unet_score_v, epoch_loss_v]))
            
            
        
            '''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            
            '''
        
        # Save Best U-Net model
            if self.save_val_best:
                if unet_score_v > best_unet_score:
                        best_unet_score = unet_score
                        best_epoch = epoch+1
                        best_unet = self.unet.state_dict()
                        print('Best %s model in epoch %d with score : %.4f '%(self.model_type,best_epoch, best_unet_score))
                        torch.save(best_unet,unet_path)
#					
            if self.stop_early:
                early_stopping(epoch_loss_v)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break 

        if not self.save_val_best:
            best_unet = self.unet.state_dict()
            torch.save(best_unet,unet_path)




        hist=np.array(hist)        
        hist = pd.DataFrame(hist) 
        hist.columns =['epoch','acc','SE','SP','PC','F1','JS','DC','unet_score', 'loss', 'acc_v','SE_v','SP_v','PC_v','F1_v','JS_v','DC_v','unet_score_v','loss_v'] 
        hist.to_csv(self.result_path + self.mode + '_training_log.csv', header=True)  
        return hist

                   
    def test(self, test_loader):
        
        self.unet.train(False)
        self.unet.eval()
        seg_array = None    
        for i, (images, GT) in enumerate(test_loader):
            images = images.to(self.device)
            SR = self.unet(images)
            SR = self.to_data(SR)
            if seg_array is None:
                seg_array = SR
            else:
                seg_array = np.concatenate((seg_array, SR))        
        seg_array = seg_array.transpose(0,2,3,1) 
        return seg_array            
			        
        
        
        
        
        

    
    
    
    
    
    