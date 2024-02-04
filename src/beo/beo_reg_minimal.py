#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl

import torchio as tio
import monai
import math
import numpy as np

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


    # code from voxelmorph repo

class NCC(nn.Module):
  """
  Local (over window) normalized cross correlation loss.
  """

  def __init__(self, win=None):
    super().__init__()
    self.win = win

  def forward(self, y_true, y_pred):

    Ii = y_true
    Ji = y_pred

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if self.win is None else self.win

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
      stride = (1)
      padding = (pad_no)
    elif ndims == 2:
      stride = (1, 1)
      padding = (pad_no, pad_no)
    else:
      stride = (1, 1, 1)
      padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)  

#from voxelmorph repo
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7):  
        super().__init__()  
        self.shape = shape
        #self.unet = monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2))
        self.unet = Unet(n_channels = 2, n_classes = 3, n_features = 32)
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)
        self.loss = NCC()
        #self.loss = nn.MSELoss()
        #self.loss = monai.losses.LocalNormalizedCrossCorrelationLoss()

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

        target = batch['target'][tio.DATA]
        source = batch['source'][tio.DATA]
        warped_source, warped_target = self(source,target)
        return self.loss(target,warped_source) + self.loss(source,warped_target)


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    args = parser.parse_args()

    subjects = []
    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
    )
    subjects.append(subject) 

    normalization = tio.ZNormalization()
    #resize = tio.Resize(128)
    #transforms = [resize, normalization]
    transforms = [normalization]
    training_transform = tio.Compose(transforms)

    training_set = tio.SubjectsDataset(subjects, transform=training_transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
#%%
    # get the spatial dimension of the data (3D)
    #in_shape = resize(subjects[0]).target.shape[1:] 
    in_shape = subjects[0].target.shape[1:]     
    reg_net = meta_registration_model(shape=in_shape)


    trainer_reg = pl.Trainer(max_epochs=args.epochs, logger=False, enable_checkpointing=False)   
    trainer_reg.fit(reg_net, training_loader)  

#%%
    # Inference
    inference_subject = training_transform(subject)
    source_data = torch.unsqueeze(inference_subject.source.data,0)
    target_data = torch.unsqueeze(inference_subject.target.data,0)    
    warped_source,warped_target = reg_net.forward(source_data,target_data)
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source.affine)
    o.save(args.output)

    #o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    #o.save('source.nii.gz')
    #o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    #o.save('target.nii.gz')    
