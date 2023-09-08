#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import glob
import multiprocessing
from tqdm import tqdm

from beo_pl_nets import DecompNet

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO Visualization')
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) initialization model', type=str, required=False, default = '/home/rousseau/Sync-Exp/Experiments/gromov_epochs_5_subj_100_patches_128_sampling_8_latentdim_10_nfilters_16_nfeatures_16.ckpt')
  parser.add_argument('-l', '--latent_dim', help='Dimension of the latent space', type=int, required=False, default = 10)
  parser.add_argument('-f', '--n_filters', help='Number of filters', type=int, required=False, default = 16)
  parser.add_argument('--n_features', help='Number of features', type=int, required=False, default = 16)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 32)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 0)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 128)
  parser.add_argument('--max_subjects', help='Max number of subjects', type=int, required=False, default = 500)
  parser.add_argument('--learning_rate', help='Learning Rate (for optimization)', type=float, required=False, default = 1e-4)

  args = parser.parse_args()

  max_subjects = args.max_subjects
  max_queue_length = args.queue
  samples_per_volume = args.samples
  num_workers = args.workers
  batch_size = args.batch_size
  patch_size = args.patch_size
  patch_overlap = int(patch_size / 2)  
  latent_dim = args.latent_dim 
  n_filters = args.n_filters
  n_features = args.n_features
  learning_rate = args.learning_rate

  data_path = home+'/Sync-Exp/Data/DHCP/'
  output_path = home+'/Sync-Exp/Experiments/'

  all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
  all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
  all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

  all_t1s = all_t1s[:max_subjects]
  max_subjects = len(all_t1s)
  subjects = []

  for t1_file in all_t1s:
      id_subject = t1_file.split('/')[6].split('_')[0:2]
      id_subject = id_subject[0]+'_'+id_subject[1]

      t2_file = [s for s in all_t2s if id_subject in s][0]
      seg_file = [s for s in all_seg if id_subject in s][0]
      
      subject = tio.Subject(
          t1=tio.ScalarImage(t1_file),
          t2=tio.ScalarImage(t2_file),
          label=tio.LabelMap(seg_file),
      )
      subjects.append(subject)

  #%%
  normalization = tio.ZNormalization(masking_method='label')
  onehot = tio.OneHot()
  crop = tio.Crop((17,17,17,17,6,5))

  validation_transform = tio.Compose([normalization, crop, onehot])

  validation_set = tio.SubjectsDataset(subjects,transform=validation_transform)
  print('Dataset size:', len(validation_set), 'subjects')

  net = DecompNet().load_from_checkpoint(args.model, latent_dim = latent_dim, n_filters = n_filters, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)
  net.eval()
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)
  net.to(device)

  Zx = []
  Zy = []

  for i in tqdm(range(len(validation_set))):
    subject = validation_set[i]
    zx = torch.zeros(1,latent_dim).to(device)
    zy = torch.zeros(1,latent_dim).to(device) 
    n = 0

    grid_sampler = tio.inference.GridSampler(
      subject,
      patch_size,
      patch_overlap,
      )

    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

    with torch.no_grad():
      for patches_batch in patch_loader:
        x_tensor = patches_batch['t1'][tio.DATA].to(device)
        y_tensor = patches_batch['t2'][tio.DATA].to(device)

        fx = net.feature_x(x_tensor)
        xfx = torch.cat([x_tensor,fx], dim=1)
        zx += net.encoder(xfx)
        
        fy = net.feature_y(y_tensor)      
        yfy = torch.cat([y_tensor,fy], dim=1)
        zy += net.encoder(yfy)

        n += 1

    zxtmp = zx/n
    zytmp = zy/n
    Zx.append(zxtmp.cpu().detach().numpy())
    Zy.append(zytmp.cpu().detach().numpy())

#%%
  import umap
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.preprocessing import StandardScaler
  
  reducer = umap.UMAP()  
  Zx_np = np.array(Zx).reshape(-1,latent_dim)
  Zy_np = np.array(Zy).reshape(-1,latent_dim)  
  Z_np = np.concatenate((Zx_np,Zy_np))
  
  Z_scaled = StandardScaler().fit_transform(Z_np)
  
  Cx = np.zeros((max_subjects,1))
  Cy = np.ones((max_subjects,1))  
  C_np = np.concatenate((Cx,Cy))
  
  embedding = reducer.fit_transform(Z_scaled)
  
  plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c = C_np[:])
  plt.gca().set_aspect('equal', 'datalim')
  plt.title('UMAP projection', fontsize=24)

#%%
  import umap.plot
  mapper = umap.UMAP().fit(Z_scaled)
  umap.plot.points(mapper, labels=C_np.reshape(-1))
  umap.plot.connectivity(mapper, edge_bundling='hammer')
  #On pourrait regarder comment est projeté une interpolation linéaire ou une géodésique dans l'espace réduit
  
  #import pandas as pd  
  #hover_data = pd.DataFrame({'index':np.arange(2*max_subjects),'label':C_np.reshape(-1)})
  #p = umap.plot.interactive(mapper, labels=C_np.reshape(-1), point_size=2)
  #umap.plot.show(p)
