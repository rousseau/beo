#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:27:20 2022

@author: rousseau
"""

import os
from os.path import expanduser
home = expanduser("~")
import nibabel
import multiprocessing
import pandas as pd

jobs = []
pool = multiprocessing.Pool(24)
output_path = home+'/Sync-Exp/Experiments/Atlas_dhcp/'
file_atlas_neonatal = home+'/Sync-Exp/Experiments/Atlas_dhcp/atlas_serag_44.nii.gz'

#atlas_neonatal = nibabel.load(file_atlas_neonatal)
data_path = '/media/rousseau/Seagate5To/Sync-Data/Cerveau/Neonatal/dhcp-rel-3/'
file_tsv = pd.read_csv(data_path+'combined.tsv', sep='\t')
n_images = file_tsv.shape[0]
for i in range(n_images):
  subject = str(file_tsv['participant_id'][i])
  session = str(file_tsv['session_id'][i])
  age = file_tsv['scan_age'][i]
    
  file_nifti = data_path+'sub-'+subject+'/ses-'+session+'/anat/sub-'+subject+'_ses-'+session+'_desc-restore_T2w.nii.gz'
  
  if os.path.exists(file_nifti):
    go = 'flirt -in '+file_nifti+' -ref '+file_atlas_neonatal+' -out '+output_path+'sub-'+subject+'_ses-'+session+'_T2w.flirt.nii.gz -omat '+output_path+'sub_'+subject+'_ses-'+session+'_flirt.txt -dof 12 '
    go+= '-searchrx -180 180 -searchry -180 180 -searchrz -180 180 -interp spline '
    print(go)
    jobs.append(go)
    
pool.map(os.system, jobs)
