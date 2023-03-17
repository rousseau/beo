#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tinycudann as tcnn
import os

class HashMLP(pl.LightningModule):
  def __init__(self, config, dim_in=3, dim_out=1):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out

    self.encoding = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
    self.mlp= tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
    self.model = torch.nn.Sequential(self.encoding, self.mlp)

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, y = batch
    z = self(x)

    loss = F.mse_loss(z, y)

    self.log("train_loss", loss)
    return loss

  def predict_step(self, batch, batch_idx):
    x, y = batch
    return self(x)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TCNN')
  parser.add_argument('-i', '--input', help='Input image (nifti)', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image (nifti)', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) trained model', type=str, required=False)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 4096)    
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default = 10)  
  parser.add_argument('-n', '--neurons', help='Number of neurons in MLP layers', type=int, required=False, default = 128)  
  parser.add_argument('-l', '--layers', help='Number of layers in MLP', type=int, required=False, default = 2)  
  parser.add_argument('-f', '--features', help='Number of features per level (hash grid)', type=int, required=False, default = 2)  
  parser.add_argument(      '--levels', help='Number of levels (hash grid)', type=int, required=False, default = 8)  
  parser.add_argument(      '--log2_hashmap_size', help='Log2 hashmap size (hash grid)', type=int, required=False, default = 15) #15:nvidia, 19: nesvor  

  args = parser.parse_args()

  image_file = args.input
  output_file = args.output
  model_file = args.model

  num_epochs = args.epochs
  batch_size = args.batch_size
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
  #https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
  config = {
  "encoding": {
		"otype": "HashGrid",
		"n_levels": args.levels,
		"n_features_per_level": args.features,
		"log2_hashmap_size": args.log2_hashmap_size,
		"base_resolution": 16,
		"per_level_scale": 1.3819#1.5
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": args.neurons,
		"n_hidden_layers": args.layers
	}
  }

  net = HashMLP(config = config, dim_in=3, dim_out=1)
  #trainer = pl.Trainer(max_epochs=num_epochs, precision=16) #3080
  trainer = pl.Trainer(gpus=1,max_epochs=num_epochs, precision=16) #Titan  
  
  trainer.fit(net, loader)

  if args.model is not None:
    trainer.save_checkpoint(model_file) 
    
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True) #remove shuffling
  yhat = torch.concat(trainer.predict(net, test_loader))

  output = np.float32(yhat.cpu().detach().numpy().reshape(data.shape))
  nib.save(nib.Nifti1Image(output, image.affine), output_file)     
