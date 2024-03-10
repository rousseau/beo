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

import monai

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7, loss='mse', lambda_img=1, lambda_mag=0, lambda_grad=0, lambda_seg=0): 
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

        self.lambda_img  = lambda_img
        self.lambda_mag  = lambda_mag
        self.lambda_grad = lambda_grad
        self.lambda_seg = lambda_seg
        #self.transformer_seg = SpatialTransformer(size=shape,mode='nearest')


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

        target = batch['target'][tio.DATA]
        source = batch['source'][tio.DATA]
        #warped_source, warped_target = self(source,target)
        x = torch.cat([source,target], dim=1)#.to(torch.float32)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)

        loss = 0

        if self.lambda_img > 0:
            warped_source = self.transformer(source, forward_flow)
            warped_target = self.transformer(target, backward_flow)
            loss_img = self.lambda_img * (self.loss(target,warped_source) + self.loss(source,warped_target))
            self.log("train_loss_img", loss_img, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += loss_img

        if self.lambda_mag > 0:  
            loss_mag = self.lambda_mag * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)  
            self.log("train_loss_mag", loss_mag, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += loss_mag

        if self.lambda_grad > 0:  
            loss_grad = self.lambda_grad * Grad3d().forward(forward_flow)  
            self.log("train_loss_grad", loss_grad, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += loss_grad

        if self.lambda_seg > 0:
            warped_source_label = self.transformer(batch['source_seg'][tio.DATA], forward_flow)
            warped_target_label = self.transformer(batch['target_seg'][tio.DATA], backward_flow)
            n_classes = batch['source_seg'][tio.DATA].shape[1] # for normalization purposes
            loss_seg = self.lambda_seg / n_classes * (F.mse_loss(warped_source_label, batch['target_seg'][tio.DATA]) + F.mse_loss(warped_target_label, batch['source_seg'][tio.DATA]))
            #loss_seg = self.lambda_seg * monai.losses.DiceCELoss(softmax=True)(warped_source_label, batch['target_seg'][tio.DATA]) + self.lambda_seg * monai.losses.DiceCELoss(softmax=True)(warped_target_label, batch['source_seg'][tio.DATA])
            self.log("train_loss_seg", loss_seg, prog_bar=True, on_epoch=True, sync_dist=True)
            loss += loss_seg    

        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, ncc, lncc)', type=str, required = False, default='mse')
    parser.add_argument('--lam_i', help='Lambda loss for image similarity', type=float, required = False, default=1)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=0)
    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    parser.add_argument('--target_seg', help='Target / Reference segmentation', type=str, required = False)
    parser.add_argument('--source_seg', help='Source / Moving segmentation', type=str, required = False)
    parser.add_argument('--lam_s', help='Lambda loss for segmentation', type=float, required = False, default=0)

    parser.add_argument('--onehot', help='Apply one hot encoding', action='store_true')

    args = parser.parse_args()

    print(args)

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

    subjects = []
    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
    )

    transforms = []
    if args.onehot:
        transforms.append(tio.transforms.OneHot())
        if args.target_seg is not None:
            subject['target_seg'] = tio.LabelMap(args.target_seg)
        if args.source_seg is not None:
            subject['source_seg'] = tio.LabelMap(args.source_seg)
    else:
        if args.target_seg is not None:
            subject['target_seg'] = tio.ScalarImage(args.target_seg)
        if args.source_seg is not None:
            subject['source_seg'] = tio.ScalarImage(args.source_seg)

    subjects.append(subject) 


    training_set = tio.SubjectsDataset(subjects,tio.Compose(transforms))    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].target.shape[1:]     
    reg_net = meta_registration_model(
        shape = in_shape, 
        loss = args.loss, 
        lambda_img = args.lam_i,
        lambda_mag = args.lam_m, 
        lambda_grad= args.lam_g,
        lambda_seg = args.lam_s)
    
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
    
    trainer_reg.fit(reg_net, training_loader)  
    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

#%%
    # Inference
    inference_subject = subject
    source_data = torch.unsqueeze(inference_subject.source.data,0)
    target_data = torch.unsqueeze(inference_subject.target.data,0)    
    warped_source,warped_target, forward_flow, backward_flow = reg_net.forward(source_data,target_data)

    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source.affine)
    o.save(args.output+'_warped.nii.gz')
    o = tio.ScalarImage(tensor=warped_target[0].detach().numpy(), affine=inference_subject.target.affine)   
    o.save(args.output+'_inverse_warped.nii.gz')
    o = tio.ScalarImage(tensor=forward_flow[0].detach().numpy(), affine=inference_subject.target.affine)   
    o.save(args.output+'_warp.nii.gz')
    o = tio.ScalarImage(tensor=backward_flow[0].detach().numpy(), affine=inference_subject.target.affine)   
    o.save(args.output+'_inverse_warp.nii.gz')
