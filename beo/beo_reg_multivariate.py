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

from beo_metrics import NCC, Grad3d
from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_loss import GetLoss

import monai

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7, loss=['mse'], lambda_loss=[1], lambda_mag=0, lambda_grad=0): 
        super().__init__()  
        self.shape = shape
        #self.unet = monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2))
        self.unet = Unet(n_channels = 2, n_classes = 3, n_features = 32)
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)

        self.loss = []
        for l in loss:
            self.loss.append(GetLoss(l))

        self.lambda_loss  = lambda_loss
        self.lambda_mag  = lambda_mag
        self.lambda_grad = lambda_grad

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
        
        target = batch['target_0'][tio.DATA]
        source = batch['source_0'][tio.DATA]

        x = torch.cat([source,target], dim=1)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)

        loss = 0
        for i in range(len(self.loss)):
            if self.lambda_loss[i] > 0:
                target = batch['target_'+str(i)][tio.DATA]
                source = batch['source_'+str(i)][tio.DATA]
                loss_img = (self.loss[i](target,self.transformer(source, forward_flow)) + self.loss[i](source,self.transformer(target, backward_flow)))
                self.log('train_loss_img_'+str(i), loss_img, prog_bar=True, on_epoch=True, sync_dist=True)
                loss += self.lambda_loss[i] * loss_img
        loss = loss / len(self.loss)        

        if self.lambda_mag > 0:  
            loss_mag = F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)  
            self.log("train_loss_mag", loss_mag, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += self.lambda_mag * loss_mag

        if self.lambda_grad > 0:  
            loss_grad = Grad3d().forward(forward_flow)  
            self.log("train_loss_grad", loss_grad, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += self.lambda_grad * loss_grad

        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Multivariate Registration 3D Image Pair')
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)

    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, action='append', required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, action='append', required = True)
    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append', required = True)
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, action='append', required = True)

    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0.1)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=1)

    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    parser.add_argument('--sigma', help='Sigma for data blurring', type=float, required = False, default=0.0)


    args = parser.parse_args()

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

    for i in range(len(args.target)):
        print('target:',args.target[i])
        print('source:',args.source[i])
        print('loss:',args.loss[i])
        print('lam_l:',args.lam_l[i])


    subjects = []
    subject = tio.Subject(
        target_0 = tio.ScalarImage(args.target[0]),
        source_0 = tio.ScalarImage(args.source[0]),
    )

    for i in range(1,len(args.target)):
        subject['target_'+str(i)] = tio.ScalarImage(args.target[i])
        subject['source_'+str(i)] = tio.ScalarImage(args.source[i])

    subjects.append(subject) 

#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].target_0.shape[1:]     
    reg_net = meta_registration_model(
        shape = in_shape, 
        loss = args.loss, 
        lambda_loss = args.lam_l,
        lambda_mag = args.lam_m, 
        lambda_grad= args.lam_g)
    
    if args.load_unet:
        reg_net.unet.load_state_dict(torch.load(args.load_unet))

    trainer_args = {
        'max_epochs' : args.epochs, 
        'strategy' : DDPStrategy(find_unused_parameters=True),
        'precision' : args.precision,
        }
    
    if args.logger is None:
        trainer_args['logger'] = False
        trainer_args['enable_checkpointing']=False
    else:    
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir = 'lightning_logs', name = args.logger)

    trainer_reg = pl.Trainer(**trainer_args)          

#%%
    # Training
    transforms = []
    if args.sigma > 0:
        epsilon = 0.001
        transforms.append(tio.transforms.RandomBlur(std=(args.sigma,args.sigma+epsilon),p=1))
    training_set = tio.SubjectsDataset(subjects,tio.Compose(transforms))    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
    
    trainer_reg.fit(reg_net, training_loader)  

    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

#%%
    # Inference
    inference_subject = subject
    source_data = torch.unsqueeze(inference_subject.source_0.data,0)
    target_data = torch.unsqueeze(inference_subject.target_0.data,0)    
    warped_source, warped_target, forward_flow, backward_flow = reg_net.forward(source_data,target_data)

    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source_0.affine)
    o.save(args.output+'_warped.nii.gz')
    o = tio.ScalarImage(tensor=warped_target[0].detach().numpy(), affine=inference_subject.target_0.affine)   
    o.save(args.output+'_inverse_warped.nii.gz')
    o = tio.ScalarImage(tensor=forward_flow[0].detach().numpy(), affine=inference_subject.target_0.affine)   
    o.save(args.output+'_warp.nii.gz')
    o = tio.ScalarImage(tensor=backward_flow[0].detach().numpy(), affine=inference_subject.target_0.affine)   
    o.save(args.output+'_inverse_warp.nii.gz')
