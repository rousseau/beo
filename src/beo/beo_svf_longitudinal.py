#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
import torch.nn as nn 

import torchio as tio
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import NCC
import monai

import math
import random

class CustomRandomDataset(Dataset):
    """
    Custom PyTorch dataset that picks 3 different random elements from another dataset.

    Args:
        dataset (torchio.Dataset): The original dataset to sample from.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return math.comb(len(self.dataset),3) # the number of unique triplets

    def __getitem__(self, idx):
        random_indices = random.sample(range(0, len(self.dataset)), 3)
        #random_indices = range(0, len(self.dataset))
        return [self.dataset[i] for i in random_indices]    
    

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
        return forward_velocity
        '''
        forward_flow = self.vecint(forward_velocity)
        warped_source = self.transformer(source, forward_flow)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        warped_target = self.transformer(target, backward_flow)

        return warped_source, warped_target
        '''
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        sample1, sample2, sample3 = batch

        #print(sample1['age'].float(),sample2['age'].float(),sample3['age'].float())

        # sample 1 (source) registered on sample 2 (target)
        svf12 = self(sample1['image'][tio.DATA],sample2['image'][tio.DATA])
        w12 = sample2['age'].float() - sample1['age'].float()        

        # sample 1 (source) registered on sample 3 (target)
        svf13 = self(sample1['image'][tio.DATA],sample3['image'][tio.DATA])
        w13 = sample3['age'].float() - sample1['age'].float()        

        #print(w12, w13)

        # constraint for linear modeling for svf
        csvf13 = w13/w12*svf12
        csvf31 = -csvf13
        flow13 = self.vecint(csvf13)
        flow31 = self.vecint(csvf31)

        csvf12 = w12/w13*svf13
        csvf21 = -csvf12
        flow12 = self.vecint(csvf12)
        flow21 = self.vecint(csvf21)

        loss12 = self.loss(sample2['image'][tio.DATA],self.transformer(sample1['image'][tio.DATA], flow12))
        loss13 = self.loss(sample3['image'][tio.DATA],self.transformer(sample1['image'][tio.DATA], flow13))
        loss21 = self.loss(sample1['image'][tio.DATA],self.transformer(sample2['image'][tio.DATA], flow21))
        loss31 = self.loss(sample1['image'][tio.DATA],self.transformer(sample3['image'][tio.DATA], flow31))
        return loss12 + loss13 + loss21 + loss31


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinaal Images')
    parser.add_argument('-f', '--filename', help='Text file containing list of nifti files and corresponding age', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, ncc, lncc)', type=str, required = False, default='ncc')
    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required = False, default=25)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required = False, default=29)

    args = parser.parse_args()

    df = pd.read_csv(args.filename, names=["nifti", "age"], sep=" ", dtype={"scalar": float})
    print(df)

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
            image=tio.ScalarImage(row["nifti"]),
            age= a * row["age"] + b
        )
        subjects.append(subject)

    normalization = tio.ZNormalization()
    transforms = [normalization]
    training_transform = tio.Compose(transforms)

    #training_set = tio.SubjectsDataset(subjects, transform=training_transform)
    training_set = tio.SubjectsDataset(subjects)    

    torch_dataset = CustomRandomDataset(training_set)
    training_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1, shuffle=True)

#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].image.shape[1:]     
    reg_net = meta_registration_model(shape=in_shape, loss=args.loss)
    if args.load_unet:
        reg_net.unet.load_state_dict(torch.load(args.load_unet))

#%%
    trainer_reg = pl.Trainer(
        max_epochs=args.epochs, 
        strategy = DDPStrategy(find_unused_parameters=True),
        logger=False, 
        enable_checkpointing=False)   
    trainer_reg.fit(reg_net, training_loader)  
    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)

    #trainer_reg.save_checkpoint('model.ckpt')
#%%
    # Inference

    # Find initial template, i.e. the one for t0
    for s in subjects:
        if s.age == 0:
            template_t0 = s
            break    

    source_data = torch.unsqueeze(template_t0.image.data,0)
    for i in range(len(subjects)):
        target_subject = training_set[i]     
        target_data = torch.unsqueeze(target_subject.image.data,0)  
        weight = target_subject.age - template_t0.age 
        svf = reg_net(source_data,target_data)
        flow = reg_net.vecint(weight*svf)
        warped_source = reg_net.transformer(source_data,flow)
        o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_subject.image.affine)
        o.save(args.output+'_svf_'+str(i+1)+'_'+args.loss+'_e'+str(args.epochs)+'.nii.gz')

    #o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    #o.save('source.nii.gz')
    #o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    #o.save('target.nii.gz')    
