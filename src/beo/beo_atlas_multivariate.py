#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import torchio as tio
import numpy as np
import random

from beo_metrics import NCC, Grad3d
from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_loss import GetLoss

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # Pick two different random indices
        idx1, idx2 = random.sample(range(len(self.dataset)), 2)
        return self.dataset[idx1],self.dataset[idx2]

    def __len__(self):
        return int(len(self.dataset)/2)
    
#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, atlas, int_steps = 7, loss=['mse'], lambda_loss=[1], lambda_mag=0, lambda_grad=0): 
        super().__init__()  
        self.shape = shape
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)

        # Network for registration between images and atlas
        self.unet_reg = Unet(n_channels = 2, n_classes = 3, n_features = 32)

        # Network for deforming the initial atlas
        #self.atlas = nn.Parameter(torch.randn(shape).unsqueeze(0).unsqueeze(0)) # Random initialization
        # atlas should be a list of 5D tensors (same shape as the input images)
        self.atlas_init = atlas 
        self.unet_atlas = Unet(n_channels = 1, n_classes = 3, n_features = 32)

        self.loss = []
        for l in loss:
            self.loss.append(GetLoss(l))

        self.lambda_loss  = lambda_loss
        self.lambda_mag  = lambda_mag
        self.lambda_grad = lambda_grad

    def forward(self,image):
        # Get first item of atlas list
        atlas_0 = self.atlas_init[0].to(image)

        forward_velocity_atlas = self.unet_atlas(atlas_0)
        forward_flow_atlas = self.vecint(forward_velocity_atlas)
        atlas_def = self.transformer(atlas_0, forward_flow_atlas)

        x = torch.cat([atlas_def,image], dim=1)
        forward_velocity = self.unet_reg(x)
        forward_flow = self.vecint(forward_velocity)
        warped_atlas = self.transformer(atlas_def, forward_flow)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        warped_image = self.transformer(image, backward_flow)

        return warped_atlas, warped_image, forward_flow, backward_flow

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch):
        # Get the images and initialize the optimizers
        tio_im1, tio_im2 = batch
        images = torch.cat([tio_im1['image_0'][tio.DATA],tio_im2['image_0'][tio.DATA]], dim=0)
        batch_size = images.shape[0]

        # Get first item of atlas list
        atlas_0 = self.atlas_init[0].to(images)
        # Prediction of the flow to deform the init atlas
        forward_velocity_atlas = self.unet_atlas(atlas_0)
        forward_flow_atlas = self.vecint(forward_velocity_atlas)
        # Deform the initial atlas
        atlas_def = self.transformer(atlas_0, forward_flow_atlas)
        # Duplicate the deformed atlas
        atlas_imgs = atlas_def.repeat(batch_size,1,1,1,1)

        # Prediction of flows between deformed atlas and pair of images
        x = torch.cat([atlas_imgs,images], dim=1)
        forward_velocity = self.unet_reg(x)
        forward_flow = self.vecint(forward_velocity)
        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)

        # Compute Losses
        loss = 0
        for i in range(len(self.loss)):
            if self.lambda_loss[i] > 0:
                images = torch.cat([tio_im1['image_'+str(i)][tio.DATA],tio_im2['image_'+str(i)][tio.DATA]], dim=0)

                atlas_i = self.atlas_init[i].to(images)
                atlas_def = self.transformer(atlas_i, forward_flow_atlas)
                atlas_imgs = atlas_def.repeat(batch_size,1,1,1,1)

                warped_atlas = self.transformer(atlas_imgs, forward_flow)
                warped_images = self.transformer(images, backward_flow)

                loss_image_space = self.loss[i](warped_atlas, images)
                loss_atlas_space = self.loss[i](warped_images, atlas_imgs)
                loss_pair_atlas_space = self.loss[i](warped_images[:int(batch_size/2)], warped_images[int(batch_size/2):])
                loss_pair_image_space = self.loss[i](warped_atlas[:int(batch_size/2)], warped_atlas[int(batch_size/2):])
                atlas_loss = loss_image_space + loss_atlas_space + loss_pair_atlas_space + loss_pair_image_space
                self.log('atlas_loss_'+str(i), atlas_loss, prog_bar=True)
                loss += self.lambda_loss[i] * atlas_loss
        loss = loss / len(self.loss)

        if self.lambda_mag > 0:  
            # Magnitude Loss for unet_reg
            loss_mag = F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)  
            self.log("train_loss_mag_unet_reg", loss_mag, prog_bar=True, on_epoch=True, sync_dist=True)
            # Magnitude Loss for unet_atlas
            loss_mag_atlas = F.mse_loss(torch.zeros(forward_flow_atlas.shape,device=self.device),forward_flow_atlas)  
            self.log("train_loss_mag_unet_atlas", loss_mag_atlas, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_mag * (loss_mag + loss_mag_atlas)

        if self.lambda_grad > 0:  
            # Gradient Loss for unet_reg
            loss_grad = Grad3d().forward(forward_flow)  
            self.log("train_loss_grad", loss_grad, prog_bar=True, on_epoch=True, sync_dist=True)
            # Gradient Loss for unet_atlas
            loss_grad_atlas = Grad3d().forward(forward_flow_atlas)
            self.log("train_loss_grad_atlas", loss_grad_atlas, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_grad * (loss_grad + loss_grad_atlas)

        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Multivariate Registration 3D Image Pair')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)

    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required = False, default=1)

    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append', required = True)
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, action='append', required = True)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0.1)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=1)

    parser.add_argument('-a', '--atlas', help='Initial Atlas', type=str, action='append', required = True)

    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    print(args)

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

#%% Load data
    df = pd.read_csv(args.tsv_file, sep='\t')

    subjects = []

    for index, row in df.iterrows():

        subject = tio.Subject(
            image_0=tio.ScalarImage(row['image']),
            image_1=tio.ScalarImage(row['onehot'])
        )
        subjects.append(subject) 

    print(len(subjects), 'subjects')

# Read atlas initialization
    atlas = []
    for i in range(len(args.atlas)):
        atlas.append(torch.unsqueeze(tio.ScalarImage(args.atlas[i]).data,0))

#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].image_0.shape[1:]     
    reg_net = meta_registration_model(
        shape = in_shape, 
        atlas = atlas,
        loss = args.loss, 
        lambda_loss = args.lam_l,
        lambda_mag = args.lam_m, 
        lambda_grad= args.lam_g)
    
    if args.load_unet:
        reg_net.unet_reg.load_state_dict(torch.load(args.load_unet))

    trainer_args = {
        'max_epochs' : args.epochs, 
        'strategy' : DDPStrategy(find_unused_parameters=True),
        'precision' : args.precision,
        'accumulate_grad_batches' : args.accumulate_grad_batches,
        }
    
    if args.logger is None:
        trainer_args['logger'] = False
        trainer_args['enable_checkpointing']=False
    else:    
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir = 'lightning_logs', name = args.logger)

    trainer_reg = pl.Trainer(**trainer_args)          

    

#%% Data loader
    training_set = tio.SubjectsDataset(subjects)
    torch_dataset = CustomDataset(training_set)
    training_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                  batch_size=1, # Pick two images only 
                                                  shuffle=True,
                                                  num_workers=4,
                                                  pin_memory=True)
#%% Training
    
    trainer_reg.fit(reg_net, training_loader)  

    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

#%%
    # Inference
    atlas_0 = reg_net.atlas_init[0].to(reg_net.device)
    forward_velocity_atlas = reg_net.unet_atlas(atlas_0)
    forward_flow_atlas = reg_net.vecint(forward_velocity_atlas)
    atlas_def = reg_net.transformer(atlas_0, forward_flow_atlas)

    o = tio.ScalarImage(tensor=atlas_def[0].detach().numpy(), affine=subjects[0].image_0.affine)
    o.save(args.output+'_atlas.nii.gz')

    '''
    for i in range(len(subjects)):
        inference_subject = subjects[i]
        image = torch.unsqueeze(inference_subject.image_0.data,0)
        warped_atlas, warped_image, forward_flow, backward_flow = reg_net.forward(image)

        o = tio.ScalarImage(tensor=warped_image[0].detach().numpy(), affine=inference_subject.image_0.affine)
        o.save(args.output+'_warped_image_'+str(i)+'.nii.gz')
    '''    
        
    

