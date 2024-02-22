
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd

import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

import random

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import NCC

import monai
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
    def __init__(self, shape, int_steps = 7, loss='ncc', loss_seg=0):  
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
        elif loss == 'mi':
            self.loss = monai.losses.GlobalMutualInformationLoss(num_bins=32)    

        if loss_seg == 1:
            self.loss_seg = monai.losses.DiceCELoss(softmax=True)      
        else:
            self.loss_seg = None    

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
        target = tio_target['image'][tio.DATA]
        source = tio_source['image'][tio.DATA]
        
        #warped_source, warped_target = self(source,target)

        x = torch.cat([source,target], dim=1)

        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)
        warped_source = self.transformer(source, forward_flow)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        warped_target = self.transformer(target, backward_flow)

        loss_img = self.loss(target,warped_source) + self.loss(source,warped_target)
        self.log("train_loss_img", loss_img, prog_bar=True, on_epoch=True)
        
        loss = loss_img

        if self.loss_seg is not None:
            warped_source_label = self.transformer(tio_source['label'][tio.DATA], forward_flow)
            warped_target_label = self.transformer(tio_target['label'][tio.DATA], backward_flow)

            loss_seg = self.loss_seg(warped_source_label, tio_target['label'][tio.DATA]) + self.loss_seg(warped_target_label, tio_source['label'][tio.DATA])

        
            self.log("train_loss_seg", loss_seg, prog_bar=True, on_epoch=True)
            loss+=loss_seg

        return loss


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Learning Registration 3D Image Pair')
    parser.add_argument('-t', '--tsv_file', help='tsv file ', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required = False, default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, ncc, lncc)', type=str, required = False, default='mse')

    parser.add_argument('--load_unet', help='Input unet model', type=str, required = False)
    parser.add_argument('--save_unet', help='Output unet model', type=str, required = True)

    parser.add_argument('--size', help='Image size', type=int, required = False, default=256)
    parser.add_argument('--seg_loss', help='Use segmentation loss', type=int, required = False, default=0)

    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required = False, default=8)


    args = parser.parse_args()

    df = pd.read_csv(args.tsv_file, sep='\t')

    subjects = []
    for index, row in df.iterrows():

        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            mask=tio.LabelMap(row['mask']),
            label=tio.LabelMap(row['label'])
        )
        subjects.append(subject) 

    print(len(subjects))

    masking = tio.Mask(masking_method='mask')
    normalization = tio.ZNormalization(masking_method='mask')    
    croporpad =tio.transforms.CropOrPad(mask_name='label')
    resize = tio.Resize(args.size)
    onehot = tio.OneHot(include=('label'))
    transforms = [masking,normalization,croporpad,resize,onehot]
    training_transform = tio.Compose(transforms)

    training_set = tio.SubjectsDataset(subjects, transform=training_transform)

    torch_dataset = CustomDataset(training_set)
    training_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=True,
                                                  num_workers=16,
                                                  pin_memory=True)

#%%
    # get the spatial dimension of the data (3D)
    in_shape = (args.size, args.size, args.size)     
    reg_net = meta_registration_model(shape=in_shape, loss=args.loss, loss_seg=args.seg_loss)
    if args.load_unet:
        reg_net.unet.load_state_dict(torch.load(args.load_unet))


#%%
    tb_logger = TensorBoardLogger("lightning_logs", name="reg_unet")


    trainer_reg = pl.Trainer(
        max_epochs=args.epochs, 
        strategy = DDPStrategy(find_unused_parameters=True),
        #precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=tb_logger)   
    
    trainer_reg.fit(reg_net, training_loader)  

    if args.save_unet:
        torch.save(reg_net.unet.state_dict(), args.save_unet)
