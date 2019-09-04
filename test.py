#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:04:16 2019

@author: rousseau
"""

#Package from OpenAI for GPU memory saving
#https://github.com/cybertronai/gradient-checkpointing
import memory_saving_gradients

from keras import backend as K
K.__dict__["gradients"] = memory_saving_gradients.gradients_memory


