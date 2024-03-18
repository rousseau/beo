#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy

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

        # Network for dynamical model of the mean trajectory
        self.unet_dyn = Unet(n_channels = 1, n_classes = 3, n_features = 32)

        # Network for deforming the initial atlas
        #self.atlas = nn.Parameter(torch.randn(shape).unsqueeze(0).unsqueeze(0)) # Random initialization
        # atlas should be a list of 5D tensors (same shape as the input images)
        self.atlas_init = atlas 
        self.unet_atlas = Unet(n_channels = 1, n_classes = 3, n_features = 32)
        
        self.learn_atlas = True
        self.learn_dyn = False

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
        # Temporal version
        flow_shape = [1,3]+list(self.shape)

        # Compute atlas at time point 0
        # Get initial loaded atlas
        atlas_0 = self.atlas_init[0].to(self.device)

        if self.learn_atlas is True : 
            # Prediction of the flow to deform the init atlas
            forward_velocity_atlas = self.unet_atlas(atlas_0)
            forward_flow_atlas = self.vecint(forward_velocity_atlas)
            # Deform the initial atlas to get atlas at time point 0
            atlas_def = self.transformer(atlas_0, forward_flow_atlas)
        else:
            atlas_def = atlas_0
            forward_flow_atlas = torch.zeros(flow_shape,device=self.device)



        # Get image pair
        tio_im1, tio_im2 = batch
        
        if self.learn_dyn is True :
            # Get dynamics 
            forward_velocity_dyn = self.unet_dyn(atlas_def)
            backward_velocity_dyn = - forward_velocity_dyn

            # Compute the atlas at time point of image 1        
            w_tp1 = tio_im1['age'].float()
            forward_flow_tp1 = self.vecint(w_tp1 * forward_velocity_dyn)
            backward_flow_tp1 = self.vecint(w_tp1 * backward_velocity_dyn)
            atlas_tp1 = self.transformer(atlas_def, forward_flow_tp1)

            # Compute the atlas at time point of image 2
            w_tp2 = tio_im2['age'].float()
            forward_flow_tp2 = self.vecint(w_tp2 * forward_velocity_dyn)
            backward_flow_tp2 = self.vecint(w_tp2 * backward_velocity_dyn)
            atlas_tp2 = self.transformer(atlas_def, forward_flow_tp2)
        else:
            atlas_tp1 = atlas_def
            atlas_tp2 = atlas_def
            forward_flow_tp1 = torch.zeros(flow_shape,device=self.device)
            forward_flow_tp2 = torch.zeros(flow_shape,device=self.device)
            backward_flow_tp1 = torch.zeros(flow_shape,device=self.device)
            backward_flow_tp2 = torch.zeros(flow_shape,device=self.device)


        # Compute the flow between image 1 and atlas at time point of image 1
        x = torch.cat([atlas_tp1, tio_im1['image_0'][tio.DATA]], dim=1)
        forward_velocity_im1 = self.unet_reg(x)
        forward_flow_im1 = self.vecint(forward_velocity_im1)        
        backward_velocity_im1 = - forward_velocity_im1
        backward_flow_im1 = self.vecint(backward_velocity_im1)

        # Compute the flow between image 2 and atlas at time point of image 2
        x = torch.cat([atlas_tp2, tio_im2['image_0'][tio.DATA]], dim=1)
        forward_velocity_im2 = self.unet_reg(x)
        forward_flow_im2 = self.vecint(forward_velocity_im2)
        backward_velocity_im2 = - forward_velocity_im2
        backward_flow_im2 = self.vecint(backward_velocity_im2)

        # Compute Losses
        loss = 0
        for i in range(len(self.loss)):
            if self.lambda_loss[i] > 0:
                
                im1 = tio_im1['image_'+str(i)][tio.DATA]
                im2 = tio_im2['image_'+str(i)][tio.DATA]

                atlas_i = self.atlas_init[i].to(self.device)
                atlas_def_i = self.transformer(atlas_i, forward_flow_atlas)

                # Deform the atlas at time point of image 1
                atlas_tp1_i = self.transformer(atlas_def_i, forward_flow_tp1)
                warped_atlas_im1 = self.transformer(atlas_tp1_i, forward_flow_im1)
                warped_image_im1 = self.transformer(im1, backward_flow_im1)

                # Deform the atlas at time point of image 2
                atlas_tp2_i = self.transformer(atlas_def_i, forward_flow_tp2)
                warped_atlas_im2 = self.transformer(atlas_tp2_i, forward_flow_im2)
                warped_image_im2 = self.transformer(im2, backward_flow_im2)

                # Losses in image space and atlas space
                loss_image_space = self.loss[i](warped_atlas_im1, im1) + self.loss[i](warped_atlas_im2, im2)
                loss_atlas_space = self.loss[i](warped_image_im1, atlas_tp1_i) + self.loss[i](warped_image_im2, atlas_tp2_i)

                # Deform images at time point 0
                warped_image_im1_t0 = self.transformer(warped_image_im1, backward_flow_tp1)
                warped_image_im2_t0 = self.transformer(warped_image_im2, backward_flow_tp2)
                # Loss in atlas space, not depending on the atlas
                loss_pair_atlas_space_t0 = self.loss[i](warped_image_im1_t0, warped_image_im2_t0) 

                # Should add loss in image space ?

                self.log('loss_image_space_'+str(i), loss_image_space, prog_bar=True)
                self.log('loss_atlas_space_'+str(i), loss_atlas_space, prog_bar=True)
                self.log('loss_pair_atlas_space_t0_'+str(i), loss_pair_atlas_space_t0, prog_bar=True)

                loss += self.lambda_loss[i] * (loss_image_space + loss_atlas_space + loss_pair_atlas_space_t0)

        loss = loss / len(self.loss)

        if self.lambda_mag > 0:
            # Magnitude Loss for unet_atlas (i.e. deformation of the initial atlas to time point t0)
            loss_mag_atlas = F.mse_loss(torch.zeros(forward_flow_atlas.shape,device=self.device),forward_flow_atlas)
            self.log("train_loss_mag_atlas", loss_mag_atlas, prog_bar=True, on_epoch=True, sync_dist=True)

            # Magnitude Loss for unet_dyn (i.e. dynamical model of the mean trajectory)
            loss_mag_dyn = F.mse_loss(torch.zeros(forward_flow_tp1.shape,device=self.device),forward_flow_tp1) 
            loss_mag_dyn+= F.mse_loss(torch.zeros(forward_flow_tp2.shape,device=self.device),forward_flow_tp2)    
            loss_mag_dyn/= 2.0 # Average over the two time points
            self.log("train_loss_mag_dyn", loss_mag_dyn, prog_bar=True, on_epoch=True, sync_dist=True)

            # Magnitude Loss for unet_reg (i.e. registration between images and atlas)
            loss_mag_reg = F.mse_loss(torch.zeros(forward_flow_im1.shape,device=self.device),forward_flow_im1)
            loss_mag_reg+= F.mse_loss(torch.zeros(forward_flow_im2.shape,device=self.device),forward_flow_im2)
            loss_mag_reg/= 2.0 # Average over the two time points
            self.log("train_loss_mag_reg", loss_mag_reg, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_mag * (loss_mag_atlas + loss_mag_dyn + loss_mag_reg)

        if self.lambda_grad > 0:
            # Gradient Loss for unet_atlas
            loss_grad_atlas = Grad3d().forward(forward_flow_atlas)
            self.log("train_loss_grad_atlas", loss_grad_atlas, prog_bar=True, on_epoch=True, sync_dist=True)

            # Gradient Loss for unet_dyn
            loss_grad_dyn = Grad3d().forward(forward_flow_tp1)
            loss_grad_dyn+= Grad3d().forward(forward_flow_tp2)
            loss_grad_dyn/= 2.0    
            self.log("train_loss_grad_dyn", loss_grad_dyn, prog_bar=True, on_epoch=True, sync_dist=True)

            # Gradient Loss for unet_reg
            loss_grad_reg = Grad3d().forward(forward_flow_im1)
            loss_grad_reg+= Grad3d().forward(forward_flow_im2)
            loss_grad_reg/= 2.0
            self.log("train_loss_grad_reg", loss_grad_reg, prog_bar=True, on_epoch=True, sync_dist=True)

            loss += self.lambda_grad * (loss_grad_atlas + loss_grad_dyn + loss_grad_reg)


        # Static version
        '''
        # Get the images
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

        # Note: lambda should be different for unet_reg and unet_atlas since one is inter-subject registration (but with close age)
        # and the other one is intra-subject registration (deformation of the atlas)
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

        '''    
        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Multivariate Registration 3D Image Pair')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)
    parser.add_argument('--max', help='Maximum age of the subjects in week', type=float, required = False, default=40)
    parser.add_argument('--min', help='Minimum age of the subjects in week', type=float, required = False, default=10)

    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required = False, default=25)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required = False, default=32)

    parser.add_argument('-a', '--atlas', help='Initial Atlas', type=str, action='append', required = True)

    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required = False, default=1)

    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append', required = True)
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, action='append', required = True)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0.1)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=1)

    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    parser.add_argument('--load_unet_atlas', help='Input unet model for atlas generation', type=str, required = False)
    parser.add_argument('--save_unet_atlas', help='Output unet model for atlas generation', type=str, required = False)
    parser.add_argument('--load_unet_dyn', help='Input unet model for dynamics', type=str, required = False)
    parser.add_argument('--save_unet_dyn', help='Output unet model for dynamics', type=str, required = False)
    parser.add_argument('--load_unet_reg', help='Input unet model for pairwise registration', type=str, required = False)
    parser.add_argument('--save_unet_reg', help='Output unet model for pairwise registration', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_gpus', help='Number of gpus (default is 0, meaning all available gpus)', type=int, required = False, default=0)

    args = parser.parse_args()
    print(args)

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

#%% Load data
    df = pd.read_csv(args.tsv_file, sep='\t')

    # Convert linearly age to [0,1] SVF interval
    # Example
    # t0 25 : 0
    # t1 29 : 1
    # ax+b -> a=1/4, b=-25/4
    a = 1.0/(args.t1-args.t0)
    b = -args.t0/(args.t1-args.t0)

    subjects = []

    for index, row in df.iterrows():

        if row["age"] < args.max and row["age"] > args.min:
            
            subject = tio.Subject(
                image_0=tio.ScalarImage(row['image']),
                image_1=tio.ScalarImage(row['onehot']),
                age= a * row["age"] + b
            )
            print(row['image'])
            print(row['onehot'])
            print(row['age'])
            print(a * row["age"] + b)

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
    
    if args.load_unet_reg:
        reg_net.unet_reg.load_state_dict(torch.load(args.load_unet_reg))
    if args.load_unet_atlas:
        reg_net.unet_atlas.load_state_dict(torch.load(args.load_unet_atlas))
    if args.load_unet_dyn:
        reg_net.unet_dyn.load_state_dict(torch.load(args.load_unet_dyn))


    trainer_args = {
        'max_epochs' : args.epochs, 
        'strategy' : DDPStrategy(find_unused_parameters=True),
        'precision' : args.precision,
        'accumulate_grad_batches' : args.accumulate_grad_batches,
        }

    if args.n_gpus > 0:
        trainer_args['devices'] = args.n_gpus

    if args.logger is None:
        trainer_args['logger'] = False
        trainer_args['enable_checkpointing']=False
    else:    
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir = 'lightning_logs', name = args.logger)

    trainer_reg = pl.Trainer(**trainer_args)          

    

#%% Data loader
    training_set = tio.SubjectsDataset(subjects)
    torch_dataset = CustomDataset(training_set)
    num_workers = args.accumulate_grad_batches * 2 # 2 workers per item in the batch
    training_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                  batch_size=1, # Pick two images only 
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
#%% Training
    
    trainer_reg.fit(reg_net, training_loader)  

    if args.save_unet_reg:
        torch.save(reg_net.unet_reg.state_dict(), args.save_unet_reg)
    if args.save_unet_atlas:
        torch.save(reg_net.unet_atlas.state_dict(), args.save_unet_atlas)
    if args.save_unet_dyn:
        torch.save(reg_net.unet_dyn.state_dict(), args.save_unet_dyn)        

#%%
    exp_name = '_'+str(args.t0)+'_'+str(args.t1)
    exp_name += '_e'+str(args.epochs)
    for i in range(len(args.loss)):
        exp_name += '_'+args.loss[i]
        exp_name += '_' + str(args.lam_l[i])
    exp_name += '_lamm'+str(args.lam_m)
    exp_name += '_lamg'+str(args.lam_g)

#%%
    # Inference
    reg_net.eval()

    # Compute the atlas at time point 0 
    atlas_0 = reg_net.atlas_init[0].to(reg_net.device)
    o = tio.ScalarImage(tensor=atlas_0[0].detach().numpy(), affine=tio.ScalarImage(args.atlas[0]).affine)
    o.save(args.output+exp_name+'_atlas_init.nii.gz')

    # Prediction of the flow to deform the init atlas
    forward_velocity_atlas = reg_net.unet_atlas(atlas_0)
    forward_flow_atlas = reg_net.vecint(forward_velocity_atlas)
    # Deform the initial atlas to get atlas at time point 0
    atlas_def = reg_net.transformer(atlas_0, forward_flow_atlas)
    o = tio.ScalarImage(tensor=atlas_def[0].detach().numpy(), affine=tio.ScalarImage(args.atlas[0]).affine)
    o.save(args.output+exp_name+'_atlas_def.nii.gz')

    # Compute atlas at different time points (-1, -0.5, 0, 0.5, 1)
    weights = torch.Tensor([-1,-0.5,0,0.5,1]).to(reg_net.device)
    forward_velocity_dyn = reg_net.unet_dyn(atlas_def)
    backward_velocity_dyn = - forward_velocity_dyn

    for w in weights:
        forward_flow_dyn = reg_net.vecint(forward_velocity_dyn*w)
        atlas_dyn = reg_net.transformer(atlas_def, forward_flow_dyn)
        o = tio.ScalarImage(tensor=atlas_dyn[0].detach().numpy(), affine=tio.ScalarImage(args.atlas[0]).affine)
        o.save(args.output+exp_name+'_atlas_'+str(w.item())+'.nii.gz')

        atlas_dyn = reg_net.transformer(atlas_0, forward_flow_dyn)
        o = tio.ScalarImage(tensor=atlas_dyn[0].detach().numpy(), affine=tio.ScalarImage(args.atlas[0]).affine)
        o.save(args.output+exp_name+'_atlas_init_'+str(w.item())+'.nii.gz')


    # Deform each subject at time point 0
    average_atlas = numpy.zeros(atlas_0.shape)
    for i in range(len(subjects)):            
        image = torch.unsqueeze(subjects[i].image_0.data,0)
        w = subjects[i].age
        print(w)
        forward_flow_dyn = reg_net.vecint(w * forward_velocity_dyn)
        atlas_dyn = reg_net.transformer(atlas_def, forward_flow_dyn)

        x = torch.cat([atlas_dyn, image], dim=1)
        forward_velocity_im = reg_net.unet_reg(x)
        backward_velocity_im = - forward_velocity_im
        backward_flow_im = reg_net.vecint(backward_velocity_im)

        warped_image = reg_net.transformer(image, backward_flow_im)

        if len(subjects) < 11:
            o = tio.ScalarImage(tensor=warped_image[0].detach().numpy(), affine=tio.ScalarImage(args.atlas[0]).affine)
            o.save(args.output+exp_name+'_warped_'+str(i)+'.nii.gz')
            o = tio.ScalarImage(tensor=image[0].detach().numpy(), affine=subjects[i].image_0.affine)
            o.save(args.output+exp_name+'_image_'+str(i)+'.nii.gz')

        average_atlas += warped_image.detach().numpy()

    average_atlas /= len(subjects)    
    print('Saving average atlas')
    o = tio.ScalarImage(tensor=average_atlas[0], affine=tio.ScalarImage(args.atlas[0]).affine)
    o.save(args.output+exp_name+'_average_atlas.nii.gz')

        
    

