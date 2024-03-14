#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F

import torchio as tio
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import Grad3d

#import math
import random
from beo_loss import GetLoss

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, atlas, int_steps = 7, loss=['mse'], lambda_loss=[1], lambda_mag=0, lambda_grad=0):  
        super().__init__()  
        self.shape = shape
        self.unet = Unet(n_channels = 1, n_classes = 3, n_features = 32)
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)

        self.atlas = atlas

        self.loss = []
        for l in loss:
            self.loss.append(GetLoss(l))

        self.lambda_loss  = lambda_loss
        self.lambda_mag  = lambda_mag
        self.lambda_grad = lambda_grad


    def forward(self,source,target):
        x = torch.cat([source,target], dim=1)
        forward_velocity = self.unet(x)
        return forward_velocity
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Extract first component of atlas
        atlas_0 = self.atlas[0].to(self.device)

        # Get the svf for the given atlas (initial point)
        forward_velocity = batch['age'].float() * self.unet(atlas_0)
        forward_flow = self.vecint(forward_velocity)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)

        loss = 0
        for i in range(len(self.loss)):
            if self.lambda_loss[i] > 0:

                # Deform the atlas 
                atlas = self.atlas[i].to(self.device)
                warped_atlas = self.transformer(atlas, forward_flow)

                # Deform the current image
                image = batch['image_'+str(i)][tio.DATA]
                warped_image = self.transformer(image, backward_flow)

                # Get the loss
                loss_i = self.loss[i](warped_atlas,image) + self.loss[i](warped_image,atlas)
                loss += self.lambda_loss[i] * loss_i

        loss = loss / len(self.loss)

        if self.lambda_mag > 0:  
            # Magnitude Loss for unet_reg
            loss_mag = F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)  
            self.log("train_loss_mag", loss_mag, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_mag * loss_mag

        if self.lambda_grad > 0:  
            # Gradient Loss for unet_reg
            loss_grad = Grad3d().forward(forward_flow)  
            self.log("train_loss_grad", loss_grad, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_grad * loss_grad

        return loss

#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required = False, default=25)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required = False, default=32)

    parser.add_argument('-a', '--atlas', help='Initial Atlas', type=str, action='append', required = True)

    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required = False, default=1)

    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append', required = True)
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, action='append', required = True)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0.1)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=1)

    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

    df = pd.read_csv(args.tsv_file, sep='\t')

    # Convert linearly age to [0,1] SVF interval
    # Example
    # t0 25 : 0
    # t1 29 : 1
    # ax+b -> a=1/4, b=-25/4
    a = 1/(args.t1-args.t0)
    b = -args.t0/(args.t1-args.t0)

    subjects = []
    for index, row in df.iterrows():
        subject = tio.Subject(
            image_0=tio.ScalarImage(row['image']),
            image_1=tio.ScalarImage(row['onehot']),
            age= a * row["age"] + b
        )
        subjects.append(subject)

    training_set = tio.SubjectsDataset(subjects)    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)

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
        reg_net.unet.load_state_dict(torch.load(args.load_unet))

#%%
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
#%%
    trainer_reg.fit(reg_net, training_loader)  
    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

#%%
    # Inference


    source_data = reg_net.atlas[0].to(reg_net.device)
    for i in range(len(subjects)):
        target_subject = training_set[i]     
        target_data = torch.unsqueeze(target_subject.image_0.data,0)  
        weight = target_subject.age 
        svf = reg_net(source_data)
        flow = reg_net.vecint(weight*svf)
        warped_source = reg_net.transformer(source_data,flow)
        o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_subject.image_0.affine)
        o.save(args.output+'_svf_'+str(i+1)+'_'+args.loss[0]+'_e'+str(args.epochs)+'.nii.gz')

    #o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    #o.save('source.nii.gz')
    #o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    #o.save('target.nii.gz')    
