#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Unet(pl.LightningModule):
    def __init__(self):
        super(Unet, self).__init__()

        self.n_channels = 1
        self.n_classes = 10
        self.n_features = 32

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )


        self.dc1 = double_conv(self.n_channels, self.n_features)
        self.dc2 = double_conv(self.n_features, self.n_features*2)
        self.dc3 = double_conv(self.n_features*2, self.n_features*4)
        self.dc4 = double_conv(self.n_features*6, self.n_features*2)
        self.dc5 = double_conv(self.n_features*3, self.n_features)

        self.mp = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.dc1(x)

        x2 = self.mp(x1)
        x2 = self.dc2(x2)

        x3 = self.mp(x2)
        x3 = self.dc3(x3)

        x4 = self.up(x3)
        x4 = torch.cat([x4,x2], dim=1)
        x4 = self.dc4(x4)

        x5 = self.up(x4)
        x5 = torch.cat([x5,x1], dim=1)
        x5 = self.dc5(x5)
        return self.out(x5)

    def training_step(self, batch, batch_idx):
        patches_batch = batch
        x = patches_batch['t2'][tio.DATA]
        y = patches_batch['seg'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('train_loss', loss)
        return loss        

    def validation_step(self, batch, batch_idx):
        patches_batch = batch
        x = patches_batch['t2'][tio.DATA]
        y = patches_batch['seg'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('val_loss', loss)
        return loss        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
