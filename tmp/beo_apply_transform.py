import numpy as np
import nibabel
import monai
import torch
import math
import torch.nn.functional as F

filename = '/home/rousseau/Sync-Exp/t2-t35.00_128.nii.gz'

#MONAI
'''
image, meta = monai.transforms.LoadImage(ensure_channel_first=True)(filename)
print(meta)

affine = monai.transforms.Affine(
    rotate_params=(0,0,0),
    scale_params=(1, 1, 1),
    translate_params=(20, 0, 0),
    shear_params=(0,0,0),
    padding_mode="zeros",
    device=torch.device("cuda:0"),
    image_only=True
)

print(image.shape)
new_img = affine(image,(128, 128,128), mode="bilinear")

monai.transforms.SaveImage(
    output_dir='/home/rousseau/',
    output_ext='.nii.gz',
    resample=False
)(new_img)
'''

target = nibabel.load(filename) #reference
source = nibabel.load(filename) #moving

target_data = torch.from_numpy(target.get_fdata()).float()
target_data = torch.reshape(target_data, (1,) + target_data.shape)
print(target_data.shape)
source_data = torch.from_numpy(source.get_fdata()).float()
source_data = torch.reshape(source_data, (1,) + source_data.shape)

target_to_world = torch.from_numpy(target.affine).to(device='cuda').float()

world_to_source = torch.from_numpy(np.linalg.inv(source.affine)).to(device='cuda').float()


translation_m = torch.eye(4).cuda()
rotation_x = torch.eye(4).cuda()
rotation_y = torch.eye(4).cuda()
rotation_z = torch.eye(4).cuda()
rotation_m = torch.eye(4).cuda()
shearing_m = torch.eye(4).cuda()
scaling_m = torch.eye(4).cuda()

trans_xyz = torch.tensor([0.1,0,0])
rotate_xyz = torch.tensor([0,0,0]) * math.pi
shearing_xyz = torch.tensor([0,0,0]) 
scaling_xyz = 1 + (torch.tensor([0,0,0]) * 0.5)

translation_m[0, 3] = trans_xyz[0]
translation_m[1, 3] = trans_xyz[1]
translation_m[2, 3] = trans_xyz[2]
scaling_m[0, 0] = scaling_xyz[0]
scaling_m[1, 1] = scaling_xyz[1]
scaling_m[2, 2] = scaling_xyz[2]

rotation_x[1, 1] = torch.cos(rotate_xyz[0])
rotation_x[1, 2] = -torch.sin(rotate_xyz[0])
rotation_x[2, 1] = torch.sin(rotate_xyz[0])
rotation_x[2, 2] = torch.cos(rotate_xyz[0])

rotation_y[0, 0] = torch.cos(rotate_xyz[1])
rotation_y[0, 2] = torch.sin(rotate_xyz[1])
rotation_y[2, 0] = -torch.sin(rotate_xyz[1])
rotation_y[2, 2] = torch.cos(rotate_xyz[1])

rotation_z[0, 0] = torch.cos(rotate_xyz[2])
rotation_z[0, 1] = -torch.sin(rotate_xyz[2])
rotation_z[1, 0] = torch.sin(rotate_xyz[2])
rotation_z[1, 1] = torch.cos(rotate_xyz[2])

rotation_m = torch.mm(torch.mm(rotation_z, rotation_y), rotation_x)

shearing_m[0, 1] = shearing_xyz[0]
shearing_m[0, 2] = shearing_xyz[1]
shearing_m[1, 2] = shearing_xyz[2]

reg_m = torch.mm(shearing_m, torch.mm(scaling_m, torch.mm(rotation_m, translation_m)))

affine_m = torch.mm(world_to_source,torch.mm(reg_m,target_to_world))

print(affine_m)

print(affine_m.shape)

grid_shape = (1,) + target_data.shape

source_data = source_data.to(device='cuda')

flow = F.affine_grid(affine_m[0:3].unsqueeze(0), grid_shape, align_corners=True)
print(flow.shape)
transformed_source = F.grid_sample(source_data.unsqueeze(0).double(), flow.double(), mode='bilinear', align_corners=True)

nibabel.save(nibabel.Nifti1Image(torch.squeeze(transformed_source).cpu().detach().numpy(), target.affine),'/home/rousseau/toto.nii.gz')    
