#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def power_method_conv(conv_model,n_filters,n_iter=5):
  n = 10
  u = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)
  v = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)

  for _ in range (n_iter):
    WTu = conv_model(torch.flip(torch.flip(u,dims=[2]),dims=[3]))
    v_new = WTu / torch.norm(WTu.flatten())
    v = v_new
            
    Wv = conv_model(v)
    u_new = Wv / torch.norm(Wv.flatten())
    u = u_new
              
  Wv = conv_model(v)
  return torch.mm( u.flatten().reshape(1,-1), Wv.flatten().reshape(-1,1)).reshape(1)

def block_lipschitz(block_x2y_list,n_filters):
  #The loss is defined as the sum of Lipschitz constant over all the mapping blocks
  n = len(block_x2y_list)
  Lipschitz_cst = torch.zeros([n]).to(device)
  n_iter = 5
  i = 0
  for blocks in block_x2y_list:
      for block in blocks:  

        cst = torch.Tensor([1]).to(device)
        j = 1
        for layername in block.state_dict().keys():
          if 'conv' in layername:
            eval('block.conv'+str(j)).weight.requires_grad = True
            conv_model = torch.nn.Sequential(
              eval('block.conv'+str(j)),
            ).to(device)

            # u = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)
            # v = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)

            # for _ in range (n_iter):
              
            #   WTu = conv_model(torch.flip(torch.flip(u,dims=[2]),dims=[3]))
            #   v_new = WTu / torch.norm(WTu.flatten())
            #   v = v_new
            
            #   Wv = conv_model(v)
            #   u_new = Wv / torch.norm(Wv.flatten())
            #   u = u_new
              
            # Wv = conv_model(v)
            # cst = cst * torch.mm( u.flatten().reshape(1,-1), Wv.flatten().reshape(-1,1)).reshape(1)
            cst = cst *  power_method_conv(conv_model,n_filters,n_iter)

            j = j+1

        Lipschitz_cst[i] = cst
        i = i+1
  return Lipschitz_cst

def compute_lipschitz_conv(weights, n_iter = 50):
  L = 0
  n_filters = weights.shape[1]
  n = 20
  u = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)
  v = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)

  conv_model = torch.nn.Sequential(
    torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, stride=1, bias=False),
  ).to(device)
  conv_model.state_dict()['0.weight'].copy_(weights)

  for i in range (n_iter):
    
    WTu = conv_model(torch.flip(torch.flip(u,dims=[2]),dims=[3]))
    v_new = WTu / torch.norm(WTu.flatten())
    v = v_new
  
    Wv = conv_model(v)
    u_new = Wv / torch.norm(Wv.flatten())
    u = u_new
    
  Wv = conv_model(v)
  L = torch.mm( u.flatten().reshape(1,-1), Wv.flatten().reshape(-1,1))
  
  return L
