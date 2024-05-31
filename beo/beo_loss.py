#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn 
import monai
from beo_metrics import NCC

import torch.nn as nn
import monai.losses

def GetLoss(loss):
  '''
  Get loss function from string

  Args:
    loss (str): The name of the loss function to retrieve. Possible values are:
      - 'mse': Mean Squared Error loss.
      - 'mae': Mean Absolute Error loss.
      - 'ncc': Normalized Cross Correlation loss.
      - 'lncc': Local Normalized Cross Correlation loss (monai).
      - 'dice': Dice loss (monai).
      - 'dicece': Dice Cross Entropy loss (monai).
      - 'mi': Global Mutual Information loss (monai).

  Returns:
    torch.nn.modules.loss._Loss: The loss function corresponding to the given name.

  Raises:
    ValueError: If the given loss function name is unknown.
  '''
  # Check the value of the loss parameter and return the corresponding loss function
  if loss == 'mse':
    return nn.MSELoss()
  elif loss == 'mae':
    return nn.L1Loss()
  elif loss == 'ncc':
    return NCC()
  elif loss == 'lncc':
    return monai.losses.LocalNormalizedCrossCorrelationLoss()
  elif loss == 'dice':
    return monai.losses.DiceLoss()
  elif loss == 'dicece':
    return monai.losses.DiceCELoss()
  elif loss == 'mi':
    return monai.losses.GlobalMutualInformationLoss(num_bins=32)
  else:
    raise ValueError(f"Unknown loss function: {loss}")

