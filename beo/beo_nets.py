#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch    
import torch.nn as nn 

# Net modules
class Unet(nn.Module):
  def __init__(self, n_channels = 2, n_classes = 3, n_features = 8):
    super(Unet, self).__init__()

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_features = n_features

    def double_conv(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
      )

    self.dc1 = double_conv(self.n_channels, self.n_features)
    self.dc2 = double_conv(self.n_features, self.n_features)
    self.dc3 = double_conv(self.n_features, self.n_features)
    self.dc4 = double_conv(self.n_features*2, self.n_features)
    self.dc5 = double_conv(self.n_features*2, self.n_features)
    
    self.ap = nn.AvgPool3d(2)

    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    self.x3_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)
    self.x4_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

    self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

  def forward(self, x):
    x1 = self.dc1(x)

    x2 = self.ap(x1)
    x2 = self.dc2(x2)

    x3 = self.ap(x2)
    x3 = self.dc3(x3)

    x4 = self.up(x3)
    x4 = torch.cat([x4,x2], dim=1)
    x4 = self.dc4(x4)

    x5 = self.up(x4)
    x5 = torch.cat([x5,x1], dim=1)
    x5 = self.dc5(x5)
    return self.out(x5)
