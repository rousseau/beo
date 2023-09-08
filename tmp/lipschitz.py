#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:41:08 2020

@author: rousseau
"""

# convert convolution kernel to matrix : 
# adapted from https://github.com/alisaaalehi/convolution_as_multiplication

#%%
import numpy as np
from scipy.linalg import toeplitz
import scipy

def matrix_to_vector(input):
    """
    Converts the input matrix to a vector by stacking the rows in a specific way explained here
    
    Arg:
    input -- a numpy matrix
    
    Returns:
    ouput_vector -- a column vector with size input.shape[0]*input.shape[1]
    """
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def convert_conv_to_matrix(I, F):
    # number of columns and rows of the input 
    I_row_num, I_col_num = I.shape 

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                               (0, output_col_num - F_col_num)),
                            'constant', constant_values=0)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    return doubly_blocked

#%%

# power method from http://mlwiki.org/index.php/Power_Iteration

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new  
  
  #%%
  
# fill I an F with random numbers
kernel = np.array([[-1, -1, -1], [0,0,0], [1,1,1]])    

n = 61
hn = int((n-1)/2)
patch = np.zeros((n,n))
patch[hn,hn] = 1  

M = convert_conv_to_matrix(patch, kernel)
lsv = np.linalg.norm(M,ord=2) #ord = 2 : 2-norm (largest sing. value)
print('using numpy.linalg.norm : '+str(lsv))

MM = np.matmul( np.transpose(M), M )
(ev,v) = power_iteration(MM)
print('using power method : '+str(np.sqrt(np.max(ev))))

import scipy.signal
ri = scipy.signal.convolve2d(patch, kernel, "same") # reponse impulsionnelle
fft_ri = np.fft.fft2(ri)
print('using fourier :'+str(np.max(np.abs(fft_ri))))

#coder la power method avec des convolutions
u = np.random.rand(n,n)
v = np.random.rand(n,n)
W = kernel
n_iter = 50
sigma = 0
sigmas = []
for i in range (n_iter):
  WTu = scipy.signal.convolve2d(np.flip(np.flip(u,axis=0),axis=1), W, "same")
  v_new = WTu / np.linalg.norm(WTu.flatten())
  v = v_new

  Wv = scipy.signal.convolve2d(v, W, "same")
  u_new = Wv / np.linalg.norm(Wv.flatten())
  u = u_new
  
  Wv = scipy.signal.convolve2d(v, W, "same")
  sigma = np.dot( np.transpose(u.flatten()), Wv.flatten())
  sigmas.append(sigma)
print('using conv power method : '+str(sigma))

import matplotlib.pyplot as plt
plt.plot(sigmas)
plt.show()

#coder la power method avec des convolutions nd
import scipy.ndimage

u = np.random.rand(n,n)
v = np.random.rand(n,n)
W = kernel
n_iter = 50
sigma = 0
sigmas = []
for i in range (n_iter):
  WTu = scipy.ndimage.convolve(np.flip(np.flip(u,axis=0),axis=1), W)
  v_new = WTu / np.linalg.norm(WTu.flatten())
  v = v_new

  Wv = scipy.ndimage.convolve(v, W)
  u_new = Wv / np.linalg.norm(Wv.flatten())
  u = u_new
  
  Wv = scipy.ndimage.convolve(v, W)
  sigma = np.dot( np.transpose(u.flatten()), Wv.flatten())
  sigmas.append(sigma)
print('using nd conv power method : '+str(sigma))


    

