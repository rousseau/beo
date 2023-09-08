# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:06:24 2019

@author: rousseau
"""

import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, BatchNormalization, Activation, Input, Add, Lambda
from keras.layers import UpSampling3D, MaxPooling3D, Subtract, GlobalAveragePooling3D, Reshape
from keras.layers import concatenate, Flatten
from keras.regularizers import l1, l2
from keras.constraints import max_norm

in1 = Input(shape=(32,24,12))	

print(K.int_shape(in1))

print(K.image_data_format())

x1 = Lambda(lambda x:x[:,:,:,:(int)(K.int_shape(in1)[-1]/2)])(in1)
print(K.int_shape(x1))
x2 = Lambda(lambda x:x[:,:,:,(int)(12/2):])(in1)
print(K.int_shape(x2))