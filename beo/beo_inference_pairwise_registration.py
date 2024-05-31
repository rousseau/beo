#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn 
import pytorch_lightning as pl

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import NCC

import monai
import torchio as tio

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7, loss='ncc'):  
        super().__init__()  
        self.shape = shape
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

        return warped_source, warped_target, forward_flow, backward_flow

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        subject1, subject2 = batch
        target = subject1['image'][tio.DATA]
        source = subject2['image'][tio.DATA]
        warped_source, warped_target = self(source,target)
        return self.loss(target,warped_source) + self.loss(source,warped_target)

#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = True)
    parser.add_argument('--size', help='Image size', type=int, required = False, default=128)

    args = parser.parse_args()

    subjects = []
    inference_target = tio.Subject(
        image=tio.ScalarImage(args.target),
    )
    inference_source = tio.Subject(
        image=tio.ScalarImage(args.source),
    )

#%%
    # get the spatial dimension of the data (3D)
    in_shape = (args.size, args.size, args.size)     
    reg_net = meta_registration_model(shape=in_shape)
    reg_net.unet.load_state_dict(torch.load(args.load_unet))

#%%
    print('Inference')

    target_data = torch.unsqueeze(inference_target.image.data,0)    
    source_data = torch.unsqueeze(inference_source.image.data,0)

    warped_source, warped_target, forward_flow, backward_flow = reg_net.forward(source_data,target_data)


#%%
    print('Saving results')
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_target.image.affine)
    o.save(args.output+'_warped.nii.gz')
    o = tio.ScalarImage(tensor=warped_target[0].detach().numpy(), affine=inference_target.image.affine)   
    o.save(args.output+'_inverse_warped.nii.gz')
    o = tio.ScalarImage(tensor=forward_flow[0].detach().numpy(), affine=inference_target.image.affine)   
    o.save(args.output+'_warp.nii.gz')
    o = tio.ScalarImage(tensor=backward_flow[0].detach().numpy(), affine=inference_target.image.affine)   
    o.save(args.output+'_inverse_warp.nii.gz')

