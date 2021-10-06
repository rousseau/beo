#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")

import torchio as tio
import glob

def get_dhcp(max_subjects=500):
  subjects = []

  data_path = home+'/Sync-Exp/Data/DHCP/'
  all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
  all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
  all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

  all_seg = all_seg[:max_subjects] 

  for seg_file in all_seg:
    id_subject = seg_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    t1_file = [s for s in all_t1s if id_subject in s][0]
      
    subject = tio.Subject(
      t2=tio.ScalarImage(t2_file),
      t1=tio.ScalarImage(t1_file),
      label=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('DHCP Dataset size:', len(dataset), 'subjects')
  return dataset