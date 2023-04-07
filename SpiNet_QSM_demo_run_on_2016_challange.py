#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:11:18 2023

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:49:39 2023

@author: venkatesh
"""

import numpy as np
import time
import scipy.io
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os
#%%

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn


from dw_WideResnet import WideResNet
from loss import *
#%%
#loading the model\
K_unrolling=3
device_id=0
#%%

input_dir='data/qsm_2016_recon_challenge/input'
output_dir='data/qsm_2016_recon_challenge/output/'
model_path='savedModels/model_weights.pth'

#%%

# loading the model..
dw=WideResNet().cuda(device_id)
dw.load_state_dict(torch.load(model_path))
dw.eval()
dw = dw.cuda(device_id)

print('#'*100)
print('loading the model......')
print('Model parameters')
print('lambda_val',dw.lambda_val.item())
print('p',dw.p.item())
print('#'*100)

#%%
# making the dipole kernel
matrix_size = [160,160, 160]
voxel_size = [1,  1,  1]

dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)

dk=dk.float().cuda(device_id)
Dk_square=torch.multiply(dk, dk)
Dk_square=Dk_square.cuda(device_id)

#%%

# loading the training stats
stats = scipy.io.loadmat(input_dir+'/tr-stats.mat')
sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)

#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")

#%%

def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon

def b_gpu(y,lambda_val, z_k):

    output1 = torch.fft.fftn(y)
    output2 = dk * output1
    output3 = torch.fft.ifftn(output2)

    w_square_z_k=w_square*z_k
    output4 = output3+lambda_val*w_square_z_k
    
    return output4

def A_gpu(x,lambda_val,p):
    output1 = Dk_square*torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    
    w_square_x=w_square*x
    output3 = output2+lambda_val * w_square_x

    return output3

def CG_GPU(local_field_gpu, z_k_gpu):

    x_0 = torch.zeros(size=(1, 1, 160,160, 160),dtype=torch.float64).cuda(device_id)
    
    temp=b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)
    
    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val,dw.p)
    p_0 = r_0

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append(torch.sum(r_old.conj()*r_old).real.item())

    for i in range(25):

        r_old_T_r_old = torch.sum(r_old.conj()*r_old)

        if(r_old_T_r_old.real.item()<1e-10):
            return x_old
        
        
        if(r_old_T_r_old.real.item()>r_stat[-1] and r_stat[-1] < 1e-06):
            return x_old


        r_stat.append( torch.sum(r_old.conj()* r_old).real.item())

        Ap_old = A_gpu(p_old,dw.lambda_val,dw.p)
        p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
        alpha = r_old_T_r_old/p_old_T_A_p_old

        # updating the x
        x_new = x_old+alpha*p_old

        # updating the remainder
        r_new = r_old-alpha*Ap_old

        # beta calculation
        r_new_T_r_new = torch.sum(r_new.conj() * r_new)

        #r_stat.append(r_new_T_r_new.real.item())
        
        beta = r_new_T_r_new/r_old_T_r_old

        # new direction p calculationubu 
        p_new = r_new+beta*p_old

        r_old = r_new
        p_old = p_new
        x_old = x_new

    return x_new


#%%
print("#"*100)
print('loading the data......')

phs=scipy.io.loadmat(input_dir+'/phs1.mat')['phs_tissue']
sus=scipy.io.loadmat(input_dir+'/cos1.mat')['cos']
msk=scipy.io.loadmat(input_dir+'/msk1.mat')['msk']
print('input dimensions are: ',phs.shape)

phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)

phs=phs.cuda(device_id)
sus=sus.cuda(device_id)
msk=msk.cuda(device_id)

print("#"*100)
#%%
with torch.no_grad():

    tic()
    print('executing the model......')
    #--------------------------------------
    # SpiNet-QSM implementation
    #--------------------------------------
    dk_repeat = dk.repeat(1,1,1,1,1)
    phs_F = torch.fft.fftn(phs,dim=[2,3,4])
    phs_F = dk_repeat * phs_F
    x_0_complex = torch.fft.ifftn(phs_F,dim=[2,3,4])
    x_0_real=z_complex_to_z_real(x_0_complex)

    x_k_real=x_0_real
    x_k_complex=x_0_complex

    
    for k in range(K_unrolling):
        x_k_complex=x_k_complex*msk
        x_k_real=z_complex_to_z_real(x_k_complex).float()
        
 
        x_k_real=(x_k_real-sus_mean)/sus_std
        z_k_real = dw(x_k_real)
        z_k_real=z_k_real*sus_std+sus_mean

        z_k_complex=z_real_to_z_complex(z_k_real)

        w=torch.pow( (x_k_complex-z_k_complex) , ((dw.p/2)-1.0))
        w_square=w.conj()*w
        w_square_sum=torch.sum(w_square)
        
        
        x_k_complex=CG_GPU(phs,z_k_complex)
        x_k_complex=x_k_complex*msk

    toc()

    x_k_cpu=(x_k_complex.real.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
    mdic  = {"susc" : x_k_cpu}
    filename  = output_dir+'/SpiNet_QSM_output.mat'
    scipy.io.savemat(filename, mdic)
    print('QSM map reconstrcuted.')
    print('Done!')
