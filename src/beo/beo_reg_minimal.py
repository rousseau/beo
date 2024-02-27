#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import torchio as tio
import numpy as np

from beo_metrics import NCC
from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet

import monai

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7, loss='mse'):  
        super().__init__()  
        self.shape = shape
        #self.unet = monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2))
        self.unet = Unet(n_channels = 2, n_classes = 3, n_features = 32)
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)
        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'ncc':                 
            self.loss = NCC()
        elif loss == 'lncc':
            self.loss = monai.losses.LocalNormalizedCrossCorrelationLoss()    


    def forward(self,source,target):
        x = torch.cat([source,target], dim=1)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)
        warped_source = self.transformer(source, forward_flow)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        warped_target = self.transformer(target, backward_flow)

        return warped_source, warped_target

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        target = batch['target'][tio.DATA]
        source = batch['source'][tio.DATA]
        warped_source, warped_target = self(source,target)
        return self.loss(target,warped_source) + self.loss(source,warped_target)


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, ncc, lncc)', type=str, required = False, default='mse')
    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    args = parser.parse_args()

    subjects = []
    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
    )
    subjects.append(subject) 

    training_set = tio.SubjectsDataset(subjects)    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].target.shape[1:]     
    reg_net = meta_registration_model(shape=in_shape, loss=args.loss)
    if args.load_unet:
        reg_net.unet.load_state_dict(torch.load(args.load_unet))

    trainer_reg = pl.Trainer(
        max_epochs=args.epochs, 
        strategy = DDPStrategy(find_unused_parameters=True),
        logger=False, 
        enable_checkpointing=False)  
    
    trainer_reg.fit(reg_net, training_loader)  
    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

#%%
    # Inference
    #inference_subject = training_transform(subject)
    inference_subject = subject
    source_data = torch.unsqueeze(inference_subject.source.data,0)
    target_data = torch.unsqueeze(inference_subject.target.data,0)    
    warped_source,warped_target = reg_net.forward(source_data,target_data)
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source.affine)
    o.save(args.output)

