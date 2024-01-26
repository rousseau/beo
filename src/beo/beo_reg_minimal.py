#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl

import torchio as tio
import monai

# Net modules
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


#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7):  
        super().__init__()  
        self.shape = shape
        self.unet = monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2))
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)
        self.loss = nn.MSELoss()

    def forward(self,target,source):
        x = torch.cat([target,source], dim=1)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)
        warped_source = self.transformer(source, forward_flow)
        return warped_source

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        target = batch['target'][tio.DATA]
        source = batch['source'][tio.DATA]
        warped_source = self(target,source)
        return self.loss(target,warped_source)


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
    resize = tio.Resize(128)
    transforms = [resize, normalization]
    training_transform = tio.Compose(transforms)

    training_set = tio.SubjectsDataset(subjects, transform=training_transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
#%%
    # get the spatial dimension of the data (3D)
    in_shape = resize(subjects[0]).target.shape[1:] 
    reg_net = meta_registration_model(shape=in_shape)


    trainer_reg = pl.Trainer(max_epochs=args.epochs,accelerator="cpu")   
    trainer_reg.fit(reg_net, training_loader)  

#%%
    # Inference
    inference_subject = resize(normalization(subjects[0]))
    source_data = torch.unsqueeze(inference_subject.source.data,0)
    target_data = torch.unsqueeze(inference_subject.target.data,0)    
    warped_source = reg_net.forward(target_data,source_data)
    print(warped_source.shape)
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source.affine)
    o.save(args.output)
