#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn 

import torchio as tio
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

from beo_nets import Unet

#%% Lightning module
class meta_image_estimation_model(pl.LightningModule):
    def __init__(self, shape): 
        super().__init__()  
        self.unet = Unet(n_channels = 1, n_classes = 1, n_features = 32)
        self.random_image = torch.randn(shape).unsqueeze(0).unsqueeze(0)
        self.loss = nn.MSELoss()

    def forward(self,x):
        pred = self.unet(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch):
        
        image = batch['image'][tio.DATA]
        noise_image = self.random_image.to(image)
        pred = self.forward(noise_image)
        loss = self.loss(pred,image)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Image Estimation')
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-i', '--image', help='Input Image', type=str, required = True)

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = False)

    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = False)

    parser.add_argument('--logger', help='Logger name', type=str, required = False, default=None)
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required = False, default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

    seed_everything(42, workers=True)

    subjects = []
    subject = tio.Subject(
        image = tio.ScalarImage(args.image),
    )
    subjects.append(subject) 

#%% Create dataloader
    training_set = tio.SubjectsDataset(subjects)    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
    
#%% Create model
    # get the spatial dimension of the data (3D)
    in_shape = subjects[0].image.shape[1:]     
    net = meta_image_estimation_model(shape = in_shape)
    
    if args.load_unet:
        net.unet_reg.load_state_dict(torch.load(args.load_unet))

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

#%% Train model
    trainer_reg.fit(net, training_loader)  

    if args.save_unet:
        torch.save(net.unet.state_dict(), args.save_unet)

#%% Inference
    inference_subject = subjects[0]
    image = torch.unsqueeze(inference_subject.image.data,0)
    pred = net.forward(image)

    o = tio.ScalarImage(tensor=pred[0].detach().numpy(), affine=inference_subject.image.affine)
    o.save(args.output)
        