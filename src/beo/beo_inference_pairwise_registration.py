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

        return warped_source, warped_target

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
    parser.add_argument('--target_mask', help='Target / Reference Mask', type=str, required = True)
    parser.add_argument('--source_mask', help='Source / Moving Mask', type=str, required = True)
    parser.add_argument('--warped_target', help='Deformed Target / Reference Image', type=str, required = True)
    parser.add_argument('--warped_source', help='Deformed Source / Moving Image', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = True)
    parser.add_argument('--size', help='Image size', type=int, required = False, default=256)

    args = parser.parse_args()

    subjects = []
    target = tio.Subject(
        image=tio.ScalarImage(args.target),
        mask=tio.LabelMap(args.target_mask),
    )
    source = tio.Subject(
        image=tio.ScalarImage(args.source),
        mask=tio.LabelMap(args.source_mask),
    )
    subjects.append(target) 
    subjects.append(source)

    masking = tio.Mask(masking_method='mask')
    normalization = tio.ZNormalization(masking_method='mask')    
    croporpad =tio.transforms.CropOrPad(mask_name='mask')
    resize = tio.Resize(args.size)
    transforms = [masking,normalization,croporpad,resize]
    composed_transform = tio.Compose(transforms)

#%%
    # get the spatial dimension of the data (3D)
    in_shape = (args.size, args.size, args.size)     
    reg_net = meta_registration_model(shape=in_shape)
    reg_net.unet.load_state_dict(torch.load(args.load_unet))

#%%
    print('Inference')
    inference_target = composed_transform(target)
    inference_source = composed_transform(source)

    target_data = torch.unsqueeze(inference_target.image.data,0)    
    source_data = torch.unsqueeze(inference_source.image.data,0)

    warped_source,warped_target = reg_net.forward(source_data,target_data)

#%%
    print('Saving results')
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_target.image.affine)
    o.save(args.warped_source)

    o = tio.ScalarImage(tensor=warped_target[0].detach().numpy(), affine=inference_source.image.affine)
    o.save(args.warped_target)

    o = tio.ScalarImage(tensor=inference_source.image.data.detach().numpy(), affine=inference_source.image.affine)
    o.save('source.nii.gz')
    o = tio.ScalarImage(tensor=inference_target.image.data.detach().numpy(), affine=inference_target.image.affine)
    o.save('target.nii.gz')    
