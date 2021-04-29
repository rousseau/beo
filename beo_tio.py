#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""
from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils import get_list_of_files
import glob
import multiprocessing

from beo_pl_nets import Unet


max_subjects = 50
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5
num_workers = 8#multiprocessing.cpu_count()

training_batch_size = 2
validation_batch_size = 1 * training_batch_size

patch_size = 128
samples_per_volume = 32
max_queue_length = 512

data_path = home+'/Sync/Data/DHCP/'
output_path = home+'/Sync/Experiments/'

all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)

all_seg = all_seg[:max_subjects]
subjects = []

for seg_file in all_seg:
    id_subject = seg_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    
    subject = tio.Subject(
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')


one_subject = dataset[0]
#one_subject.plot()
print(one_subject)
print(one_subject.t2)
print(one_subject.seg)

#%%

training_transform = tio.Compose([
  #tio.ToCanonical(),
  #tio.RandomMotion(p=0.2),
  #tio.RandomBiasField(p=0.3),
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
  #tio.RandomNoise(p=0.5),
  #tio.RandomFlip(),
  #tio.OneOf({
  #  tio.RandomAffine(): 0.8,
  #  tio.RandomElasticDeformation(): 0.2,
  #}),
  tio.OneHot(),
])

validation_transform = tio.Compose([
  #tio.ToCanonical(),
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
  tio.OneHot(),
])

#%%
seed = 42  # for reproducibility


num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')

#%%

#training_instance = training_set[0]  # transform is applied inside SubjectsDataset
#training_instance.plot()
#print(training_instance['seg'].shape)
#training_instance['seg'].save('toto.nii.gz')

#%%
print('num_workers : '+str(num_workers))

probabilities = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
#sampler = tio.data.UniformSampler(patch_size)
sampler = tio.data.LabelSampler(
          patch_size=patch_size,
          label_name='seg',
          label_probabilities=probabilities,
)

patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=False,
    shuffle_patches=False,
)

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

#%%
# import torchvision
# from IPython import display

# one_batch = next(iter(training_loader_patches))
# k = int(patch_size // 4)
# batch_mri = one_batch['mri'][tio.DATA][..., k]
# batch_label = one_batch['brain'][tio.DATA][:, 1:, ..., k]
# slices = torch.cat((batch_mri, batch_label))
# image_path = 'batch_patches.png'
# torchvision.utils.save_image(
#     slices,
#     image_path,
#     nrow=training_batch_size,
#     normalize=True,
#     scale_each=True,
# )
# display.Image(image_path)

#%%
#repo = 'fepegar/highresnet'
#model_name = 'highres3dnet'
#model = torch.hub.load(repo, model_name, pretrained=True)

#model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#    in_channels=1, out_channels=1, init_features=32, pretrained=True)

#from highresnet import HighRes3DNet
#model = HighRes3DNet(in_channels=1, out_channels=10)



net = Unet()


#%%
trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20)
trainer.fit(net, training_loader_patches)

print('Finished Training')


#%%
print('Inference')

patch_overlap = 4, 4, 4  # or just 4
patch_size = 64, 64, 64

subject = validation_set[0]

grid_sampler = tio.inference.GridSampler(
  subject,
  patch_size,
  patch_overlap,
  )

patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
aggregator = tio.inference.GridAggregator(grid_sampler)
net.eval()

with torch.no_grad():
  for patches_batch in patch_loader:
    input_tensor = patches_batch['t2'][tio.DATA]
    locations = patches_batch[tio.LOCATION]
    outputs = net(input_tensor)
    #logits = net(input_tensor)
    #labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
    #outputs = labels
    aggregator.add_batch(outputs, locations)
output_tensor = aggregator.get_output_tensor()

print(output_tensor.shape)

output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['t2'].affine)
output_seg.save(output_path+'toto.nii.gz')
subject['t2'].save(output_path+'toto_t2.nii.gz')
subject['seg'].save(output_path+'toto_seg.nii.gz')
trainer.save_checkpoint(output_path+'example_model.ckpt')

print(output_seg.affine)
print(subject['t2'].affine)
print(subject['seg'].affine)