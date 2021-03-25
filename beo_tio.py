#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""

import torch
import torchio as tio
from torch.utils.data import DataLoader

from utils import get_list_of_files
import glob

data_path = '/media/rousseau/Seagate5To/Cerveau/Neonatal/dhcp-rel-2/dhcp_anat_pipeline/'

all_seg = glob.glob(data_path+'**/*desc-ribbon_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)

subjects = []

for seg_file in all_seg:
    id_subject = seg_file.split('/')[8]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    
    subject = tio.Subject(
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')


one_subject = dataset[0]
one_subject.plot()
print(one_subject)
print(one_subject.t2)
print(one_subject.seg)