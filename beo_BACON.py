#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")
import nibabel as nib
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import math 
import numpy as np

#Code adapted from Diego's code
class ResFourierFilter(nn.Module):
  def __init__(self, dim_in=3, dim_hidden=128, bias=True, Bmin=0, Bwidth=16):
    super().__init__()

    self.Bmin = Bmin
    self.Bwidth = Bwidth

    self.linear = torch.nn.Linear(dim_in, dim_hidden, bias=bias)
    
    r_max = Bmin + Bwidth
    r_min = Bmin
    theta = torch.rand(dim_hidden)*2*math.pi
    phi   = torch.rand(dim_hidden)*2*math.pi
    # uniform sampling of radious \in [r_min, r_max]
    A = 2/(r_max*r_max - r_min*r_min)
    r = torch.sqrt(2*torch.rand(dim_hidden)/A + r_min*r_min)

    gamma = torch.zeros([dim_hidden,dim_in])

    #uniform sampling in a sphere
    #https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    phi = torch.rand(dim_hidden)*2*math.pi
    costheta = 2*(torch.rand(dim_hidden)-0.5)
    u = torch.rand(dim_hidden)

    theta = torch.arccos(costheta)
    r = torch.pow(u, 1/3)*(r_max-r_min) + r_min

    gamma[:,0] = r * torch.sin(theta) * torch.cos(phi)
    gamma[:,1] = r * torch.sin(theta) * torch.sin(phi)
    gamma[:,2] = r * torch.cos(theta)


    # Init weights
    self.linear.weight.data = gamma
    self.linear.bias.data.uniform_(-math.pi, math.pi)

  def forward(self, x):
    return torch.sin(self.linear(x))

class ResMFN(pl.LightningModule):
  def __init__(self, dim_in=3, dim_hidden=128, dim_out=1):
    super().__init__()

    filter_fun = ResFourierFilter

    self.g0 = filter_fun(dim_in, dim_hidden, Bmin=0, Bwidth=8) # 8
    self.g1 = filter_fun(dim_in, dim_hidden, Bmin=8, Bwidth=8) # 16
    self.g2 = filter_fun(dim_in, dim_hidden, Bmin=16, Bwidth=16) # 32
    self.g3 = filter_fun(dim_in, dim_hidden, Bmin=32, Bwidth=32) # 64
    self.g4 = filter_fun(dim_in, dim_hidden, Bmin=64, Bwidth=64) # 128

    self.l1 = torch.nn.Linear(dim_hidden, dim_hidden)
    self.l2 = torch.nn.Linear(dim_hidden, dim_hidden)
    self.l3 = torch.nn.Linear(dim_hidden, dim_hidden)
    self.l4 = torch.nn.Linear(dim_hidden, dim_hidden)

    self.y1 = torch.nn.Linear(dim_hidden, dim_out)
    self.y2 = torch.nn.Linear(dim_hidden, dim_out)
    self.y3 = torch.nn.Linear(dim_hidden, dim_out)
    self.y4 = torch.nn.Linear(dim_hidden, dim_out)

    factor = torch.sqrt(torch.as_tensor(6.0 / dim_hidden))

    self.l1.weight.data.uniform_(-factor, factor)
    self.l2.weight.data.uniform_(-factor, factor)
    self.l3.weight.data.uniform_(-factor, factor)
    self.l4.weight.data.uniform_(-factor, factor)

    self.y1.weight.data.uniform_(-factor, factor)
    self.y2.weight.data.uniform_(-factor, factor)
    self.y3.weight.data.uniform_(-factor, factor)
    self.y4.weight.data.uniform_(-factor, factor)

  def forward(self, xin):
    z0 = self.g0(xin)

    z1 = self.l1(z0)*self.g1(xin)
    z2 = self.l2(z1)*self.g2(xin)
    z3 = self.l3(z2)*self.g3(xin)
    z4 = self.l4(z3)*self.g4(xin)

    y1 = self.y1(z1)
    y2 = y1 + self.y2(z2)
    y3 = y2 + self.y3(z3)
    y4 = y3 + self.y4(z4)

    return y4, y3, y2, y1

  def training_step(self, batch, batch_idx):
    x,y = batch    
    z = self(x)[0] #keep only y4 output

    loss = F.mse_loss(z, y)
    return loss

  def predict_step(self, batch, batch_idx):
    x,y = batch    
    return self(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo SIREN')
  parser.add_argument('-i', '--input', help='Input image (nifti)', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image (nifti)', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) trained model', type=str, required=False)
  parser.add_argument('-n', '--neurons', help='Number of hidden neurons', type=int, required=False, default = 512)
  parser.add_argument('-l', '--layers', help='Number of layers', type=int, required=False, default = 5)  
  parser.add_argument('-w', '--w0', help='Value of w_0', type=float, required=False, default = 30)  
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default = 10)  
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1024)    
  parser.add_argument('--workers', help='Number of workers (multiprocessing). By default: the number of CPU', type=int, required=False, default = -1)
  parser.add_argument("--accelerator", default='gpu')

  args = parser.parse_args()

  dim_hidden = args.neurons
  num_epochs = args.epochs
  model_file = args.model
  batch_size = args.batch_size
  image_file = args.input
  output_file = args.output
  num_workers = args.workers
  accelerator = args.accelerator

  if num_workers == -1:
    num_workers = os.cpu_count()
  
  #Read image
  image = nib.load(image_file)
  data = image.get_fdata()

  #Create grid
  dim = 3
  x = torch.linspace(-1, 1, steps=data.shape[0])
  y = torch.linspace(-1, 1, steps=data.shape[1])
  z = torch.linspace(-1, 1, steps=data.shape[2])
  
  mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
  
  #Convert to X=(x,y,z) and Y=intensity
  X = torch.Tensor(mgrid.reshape(-1,dim))
  Y = torch.Tensor(data.flatten())
  
  #Normalize intensities between [-1,1]
  Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) * 2 - 1
  Y = torch.reshape(Y, (-1,1))
  
  #Pytorch dataloader
  dataset = torch.utils.data.TensorDataset(X,Y)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

  #Training
  net = ResMFN(dim_in=3, dim_hidden=dim_hidden, dim_out=1)
  trainer = pl.Trainer(accelerator=accelerator, max_epochs=num_epochs, precision=16)
  trainer.fit(net, loader)

  if args.model is not None:
    trainer.save_checkpoint(model_file) 
    
  #%% Load trained model (just to check that loading is working) and do the prediction using lightning trainer (for batchsize management)
  #net = SirenNet().load_from_checkpoint(model_file, dim_in=3, dim_hidden=dim_hidden, dim_out=1, num_layers=num_layers, w0 = w0)

  batch_size_test = batch_size * 2 
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, num_workers=num_workers, pin_memory=True) #remove shuffling
  
  yhat = trainer.predict(net, test_loader)

  yhat_4 = torch.cat([x[0] for x in yhat])  
  output = np.float32(yhat_4.cpu().detach().numpy().reshape(data.shape))
  nib.save(nib.Nifti1Image(output, image.affine), output_file)     

  nib.save(nib.Nifti1Image(np.float32(torch.cat([x[0] for x in yhat]).cpu().detach().numpy().reshape(data.shape)), image.affine), 'y4_'+output_file)     
  nib.save(nib.Nifti1Image(np.float32(torch.cat([x[1] for x in yhat]).cpu().detach().numpy().reshape(data.shape)), image.affine), 'y3_'+output_file)     
  nib.save(nib.Nifti1Image(np.float32(torch.cat([x[2] for x in yhat]).cpu().detach().numpy().reshape(data.shape)), image.affine), 'y2_'+output_file)     
  nib.save(nib.Nifti1Image(np.float32(torch.cat([x[3] for x in yhat]).cpu().detach().numpy().reshape(data.shape)), image.affine), 'y1_'+output_file)     
