#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import math
import numpy as np


#Code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
import numbers

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = int((kernel_size-1)/2)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


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

class Grad3d:
  """
  3-D gradient loss.
  """
  def __init__(self, penalty='l1'):
    self.penalty = penalty

  def forward(self, y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    if self.penalty == 'l2':
      dy = dy * dy
      dx = dx * dx
      dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    return grad 

class NCC:
  """
  Local (over window) normalized cross correlation loss.
  """

  def __init__(self, win=None):
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

class MSE:
  """
  Mean squared error loss.
  """

  def forward(self, y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


class registration_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7, n_features = 8):  
    super().__init__()  
    self.shape = shape

    self.unet_model = Unet(n_features = n_features)
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)

    self.lambda_similarity = 1
    self.lambda_grad_flow  = 0
    self.lambda_magn_flow  = 0
    self.lambda_grad_velocity  = 0.1
    self.lambda_magn_velocity  = 0.01
    self.bidir = True
    self.similarity = monai.losses.LocalNormalizedCrossCorrelationLoss() #monai.losses.GlobalMutualInformationLoss(num_bins=32) #NCC() #MSE()

  def forward(self,source,target):
    #concatenate images for unet
    x = torch.cat([source,target],dim=1)
    forward_velocity = self.unet_model(x)
    
    backward_velocity = -forward_velocity
    if self.int_steps > 0:
      forward_flow = self.vecint(forward_velocity)
      backward_flow= self.vecint(backward_velocity) 
    else:
      forward_flow = forward_velocity
      backward_flow= backward_velocity
    
    y_source = self.transformer(source, forward_flow)
    y_target = self.transformer(target, backward_flow)
    
    return y_source, y_target, forward_velocity, forward_flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch

    y_source,y_target, forward_velocity, forward_flow = self(source,target)
    
    if self.bidir is True:
      loss = self.lambda_similarity * (self.similarity.forward(target,y_source) + self.similarity.forward(y_target,source))/2
    else:
      loss = self.lambda_similarity * self.similarity.forward(target,y_source)
      
    if self.lambda_grad_flow > 0:
      loss += self.lambda_grad_flow * Grad3d().forward(forward_flow) 

    if self.lambda_magn_flow > 0:  
      loss += self.lambda_magn_flow * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)    
    
    if self.lambda_grad_velocity > 0:
      loss += self.lambda_grad_velocity * Grad3d().forward(forward_velocity) 
    if self.lambda_magn_velocity > 0:  
      loss += self.lambda_magn_velocity * F.mse_loss(torch.zeros(forward_velocity.shape,device=self.device),forward_velocity)  
    return loss 
