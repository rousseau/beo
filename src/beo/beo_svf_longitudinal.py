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
from beo_metrics import Grad3d

#import math
import random
from beo_loss import GetLoss

class CustomRandomDataset(Dataset):
    """
    Custom PyTorch dataset that picks 3 different random elements from another dataset.

    Args:
        dataset (torchio.Dataset): The original dataset to sample from.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        #return math.comb(len(self.dataset),3) # the number of unique triplets
        return len(self.dataset)-2 # the number of triplets with two anchor points

    def __getitem__(self, idx):
        #random_indices = random.sample(range(0, len(self.dataset)), 3)
        indice_t0 = 0
        indice_t1 = 0
        for d in range(len(self.dataset)):
            if self.dataset[d]['age'] == 0:
                indice_t0 = d
            if self.dataset[d]['age'] == 1:
                indice_t1 = d

        indice_random = random.randint(0, len(self.dataset)-1)
        if indice_random == indice_t0 or indice_random == indice_t1:
            indice_random = (indice_random + 1) % len(self.dataset)

        anchors = [indice_t0, indice_t1, indice_random]     
        return [self.dataset[i] for i in anchors]
        #return [self.dataset[i] for i in random_indices]    
    

#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7, loss=['mse'], lambda_loss=[1], lambda_mag=0, lambda_grad=0):  
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
        svf12 = self(sample1['image_0'][tio.DATA],sample2['image_0'][tio.DATA])
        w12 = sample2['age'].float() - sample1['age'].float()        

        # sample 1 (source) registered on sample 3 (target)
        svf13 = self(sample1['image_0'][tio.DATA],sample3['image_0'][tio.DATA])
        w13 = sample3['age'].float() - sample1['age'].float()        

        #print(w12, w13)

        # constraint for linear modeling for svf using random anchors
        #csvf13 = w13/w12*svf12
        #csvf12 = w12/w13*svf13

        # constraint for linear modeling using fixed chosen anchors
        csvf12 = svf12
        csvf13 = w13*svf12


        csvf31 = -csvf13
        flow13 = self.vecint(csvf13)
        flow31 = self.vecint(csvf31)

        csvf21 = -csvf12
        flow12 = self.vecint(csvf12)
        flow21 = self.vecint(csvf21)

        loss = 0
        loss12 = self.loss[0](sample2['image_0'][tio.DATA],self.transformer(sample1['image_0'][tio.DATA], flow12))
        loss13 = self.loss[0](sample3['image_0'][tio.DATA],self.transformer(sample1['image_0'][tio.DATA], flow13))
        loss21 = self.loss[0](sample1['image_0'][tio.DATA],self.transformer(sample2['image_0'][tio.DATA], flow21))
        loss31 = self.loss[0](sample1['image_0'][tio.DATA],self.transformer(sample3['image_0'][tio.DATA], flow31))
        loss += loss12 + loss13 + loss21 + loss31

        if self.lambda_mag > 0:  
            loss_mag12 = F.mse_loss(torch.zeros(flow12.shape,device=self.device),flow12)  
            loss_mag13 = F.mse_loss(torch.zeros(flow13.shape,device=self.device),flow13)  
            loss_mag21 = F.mse_loss(torch.zeros(flow21.shape,device=self.device),flow21)  
            loss_mag31 = F.mse_loss(torch.zeros(flow31.shape,device=self.device),flow31)  
            loss_mag = self.lambda_mag * (loss_mag12 + loss_mag13 + loss_mag21 + loss_mag31) / 4
            loss += loss_mag

        if self.lambda_grad > 0:  
            loss_grad12 = Grad3d().forward(flow12)  
            loss_grad13 = Grad3d().forward(flow13)  
            loss_grad21 = Grad3d().forward(flow21)  
            loss_grad31 = Grad3d().forward(flow31)  
            loss_grad = self.lambda_grad * (loss_grad12 + loss_grad13 + loss_grad21 + loss_grad31) / 4
            loss += loss_grad

        return loss



#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required = False, default=25)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required = False, default=29)

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
    torch_dataset = CustomRandomDataset(training_set)
    training_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1)

#%%
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].image_0.shape[1:]     

    reg_net = meta_registration_model(
        shape = in_shape, 
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

    #trainer_reg.save_checkpoint('model.ckpt')
#%%
    # Inference

    # Find initial template, i.e. the one for t0
    for s in subjects:
        if s.age == 0:
            template_t0 = s
            break    

    source_data = torch.unsqueeze(template_t0.image_0.data,0)
    for i in range(len(subjects)):
        target_subject = training_set[i]     
        target_data = torch.unsqueeze(target_subject.image_0.data,0)  
        weight = target_subject.age - template_t0.age 
        svf = reg_net(source_data,target_data)
        flow = reg_net.vecint(weight*svf)
        warped_source = reg_net.transformer(source_data,flow)
        o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_subject.image_0.affine)
        o.save(args.output+'_svf_'+str(i+1)+'_'+args.loss[0]+'_e'+str(args.epochs)+'.nii.gz')

    #o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    #o.save('source.nii.gz')
    #o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    #o.save('target.nii.gz')    
