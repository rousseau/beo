
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
from pytorch_lightning.loggers import TensorBoardLogger

import random

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import Grad3d
from beo_loss import GetLoss

import torchio as tio


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
    def __init__(self, shape, int_steps = 7, loss=['ncc'], lambda_loss=[1], lambda_mag=0, lambda_grad=0):  
        super().__init__()  
        self.shape = shape
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

        return warped_source, warped_target

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        tio_target, tio_source = batch
        target = tio_target['image_0'][tio.DATA]
        source = tio_source['image_0'][tio.DATA]

        x = torch.cat([source,target], dim=1)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)

        loss = 0
        for i in range(len(self.loss)):
            if self.lambda_loss[i] > 0:
                target = tio_target['image_'+str(i)][tio.DATA]
                source = tio_source['image_'+str(i)][tio.DATA]
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
    parser = argparse.ArgumentParser(description='Beo dHCP Learning Registration 3D Image Pair')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)

    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required = False, default=1)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required = False, default=8)

    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append', required = True)

    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, action='append', required = True)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required = False, default=0.1)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required = False, default=1)


    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = True)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    print(args)
#%% Load data
    df = pd.read_csv(args.tsv_file, sep='\t')

    subjects = []
    for index, row in df.iterrows():

        subject = tio.Subject(
            image_0=tio.ScalarImage(row['image']),
            image_1=tio.LabelMap(row['onehot'])
        )
        subjects.append(subject) 

    print(len(subjects), 'subjects')

#%% Data loader
    training_set = tio.SubjectsDataset(subjects)
    torch_dataset = CustomDataset(training_set)
    training_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=True,
                                                  num_workers=16,
                                                  pin_memory=True)

#%% Model
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].image_0.shape[1:]     

    reg_net = meta_registration_model(
        shape = in_shape, 
        loss = args.loss, 
        lambda_loss = args.lam_l,
        lambda_mag = args.lam_m, 
        lambda_grad= args.lam_g
        )

    if args.load_unet:
        reg_net.unet.load_state_dict(torch.load(args.load_unet))


#%% Lightning trainer
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

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

#%% Training
    
    trainer_reg.fit(reg_net, training_loader)  

    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)
