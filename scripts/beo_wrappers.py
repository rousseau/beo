#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
import getpass
import shutil
import tempfile
import nibabel as nib

def wrapper_scunet(input: str, output: str, noise: int = None) -> None:
    print('Denoising using SCUNet')
    home = expanduser("~")
    cmd_line = 'time python '+os.path.join(home, 'Sync-Exp','SCUNet','main_3dmri.py')
    cmd_line+= ' --model_zoo '+os.path.join(home, 'Sync-Exp','SCUNet','model_zoo')
    cmd_line+= ' -i '+input
    cmd_line+= ' -o '+output
    if noise is not None:
        cmd_line+= ' -n '+str(noise)
    print(cmd_line)
    os.system(cmd_line)    

### Masking methods

def wrapper_masking(input: str, output: str, method: str) -> None:
    """
    Wrapper for brain masking
    input: input image (nifti)
    output: output image (nifti)
    method: masking method (nesvor, niftymic, synthstrip)
    """
    if method == "nesvor":
        wrapper_nesvor_masking(input, output)
    elif method == 'niftymic':
        wrapper_niftymic_masking(input, output)
    elif method == 'synthstrip':
        wrapper_synthstrip_masking(input, output)
    else:
        print('Method not implemented')
        return

def wrapper_nesvor_masking(input: str, output: str) -> None:
    print('Brain masking using Nesvor')
    input_directory = os.path.dirname(input)
    input_file = os.path.basename(input)
    output_directory = os.path.dirname(output)
    output_file = os.path.basename(output)
    cmd_line = 'time docker run --rm --gpus all --ipc=host '
    cmd_line += '-v ' + input_directory + ':/incoming:ro '
    cmd_line += '-v ' + output_directory + ':/outgoing:rw '
    cmd_line += 'junshenxu/nesvor '
    cmd_line += 'nesvor segment-stack '
    cmd_line += '--input-stacks /outgoing/' + input_file + ' '
    cmd_line += '--output-stack-masks /outgoing/' + output_file + ' '
    print(cmd_line)
    os.system(cmd_line)

def wrapper_niftymic_masking(input: str, output: str) -> None:
    print('Brain masking using niftyMIC')
    username = getpass.getuser() # assuming that the home directory is /home/username
    home = expanduser("~")
    cmd_line = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_segment_fetal_brains '
    cmd_line+= ' --filenames '+input.replace(username,'data')+' --filenames-masks '+output.replace(username,'data')
    print(cmd_line)
    os.system(cmd_line)    

def wrapper_synthstrip_masking(input: str, output: str) -> None:
    print('Brain masking using mri_synthstrip')
    cmd_line = 'mri_synthstrip -i '+input+' -m '+output
    print(cmd_line)
    os.system(cmd_line)

### Reconstruction methods

def wrapper_reconstruction(input: str, mask: str, output: str, method: str) -> None:
    '''
    Wrapper for reconstruction
    input: list of input image (nifti)
    mask: list of mask image (nifti)
    output: output image (nifti)
    method: reconstruction method (nesvor, niftymic, svrtk)   
    '''
    if method == 'nesvor':
        wrapper_nesvor_reconstruction(input, mask, output)
    elif method == 'niftymic':
        wrapper_niftymic_reconstruction(input, mask, output)
    elif method == 'svrtk':
        wrapper_svrtk_reconstruction(input, mask, output)
    else:
        print('Method not implemented')
        return    

def wrapper_nesvor_reconstruction(input: str, mask: str, output: str) -> None:
    print('Reconstruction using nesvor')
    input_directory = os.path.dirname(input[0])
    output_directory = os.path.dirname(output)
    output_file = os.path.basename(output)
    cmd_line = 'time docker run --rm --gpus all --ipc=host '
    cmd_line += '-v ' + input_directory + ':/incoming:ro '
    cmd_line += '-v ' + output_directory + ':/outgoing:rw '
    cmd_line += 'junshenxu/nesvor '
    cmd_line += 'nesvor reconstruct '
    for file in input:
        cmd_line += '--input-stacks /incoming/' + os.path.basename(file) + ' '
    for mask_stack in mask:
        cmd_line += '--stack-masks /incoming/' + os.path.basename(mask_stack) + ' '

    cmd_line += '--bias-field-correction --output-resolution 6 ' 
    cmd_line += '--output-volume '

    cmd_line += '/outgoing/nesvor_r6.nii.gz '
    cmd_line += '--output-model /outgoing/model_nesvor.pt '

    print(cmd_line)
    os.system(cmd_line)

    cmd_line = 'time docker run --rm --gpus all --ipc=host '
    cmd_line += '-v ' + input_directory + ':/incoming:ro '
    cmd_line += '-v ' + output_directory + ':/outgoing:rw '
    cmd_line += 'junshenxu/nesvor '
    cmd_line += 'nesvor sample-volume --inference-batch-size 2048 --verbose 2 --output-volume '
    cmd_line += '/outgoing/'+output_file
    cmd_line += ' --output-resolution 0.5 '
    cmd_line += '--input-model /outgoing/model_nesvor.pt '

    print(cmd_line)
    os.system(cmd_line)        
    try:
        os.remove(os.path.join(output_directory, 'nesvor_r6.nii.gz'))
    except FileNotFoundError:
        print(f"File {'nesvor_r6.nii.gz'} not found, skipping removal.")

def wrapper_niftymic_reconstruction(input: str, mask: str, output: str) -> None:
    print('Reconstruction using niftymic')
    username = getpass.getuser() # assuming that the home directory is /home/username
    home = expanduser("~")
    cmd_line = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_reconstruct_volume '
    docker_output = os.path.dirname(output)
    docker_output = docker_output.replace(username,'data')        
    cmd_line+= ' --output '+os.path.join(docker_output,os.path.basename(output))+' --isotropic-resolution 0.5 '

    cmd_line+= ' --filenames '
    docker_stacks = [s.replace(username,'data') for s in input]
    docker_masks = [s.replace(username,'data') for s in mask]

    for i in docker_stacks:
        cmd_line+= i+' '
    cmd_line+= ' --filenames-masks '
    for i in docker_masks:
        cmd_line+= i+' '

    print(cmd_line)
    os.system(cmd_line)    

def wrapper_svrtk_reconstruction(input: str, mask: str, output: str) -> None:
    print('Reconstruction using SVRTK')
    username = getpass.getuser() # assuming that the home directory is /home/username
    home = expanduser("~")

    output_directory = os.path.dirname(output)
    output_file = os.path.basename(output)

    # Create a temporary directory to store the input files for svrtk
    with tempfile.TemporaryDirectory(dir=home) as temp_dir:
        # temp_dir is the path to the temporary directory
        print(f'Temporary directory: {temp_dir}')

        # Copy files into the temporary directory
        for f in input:
            shutil.copy(f, temp_dir)

        svrtk_input = temp_dir.replace(username,'data')
        svrtk_output = output_directory.replace(username,'data')

        slice_thickness = max(nib.load(input[0]).header['pixdim'])
        print('Slice thickness:',slice_thickness)

        cmd_line = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
        cmd_line+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-reconstruction.sh '+svrtk_input+' '+svrtk_output+' 1 '+str(slice_thickness)+' 0.5 1'
        print(cmd_line)
        os.system(cmd_line)  
        os.rename(os.path.join(output_directory,'reo-SVR-output-brain.nii.gz'), os.path.join(output_directory,output_file))  

### SVRTK wrappers

def wrapper_svrtk_reorientation(input, output) -> None:
    home = expanduser("~")
    username = getpass.getuser() # assuming that the home directory is /home/username
    input_file = os.path.basename(input)
    output_directory = os.path.dirname(output)
    output_file = os.path.basename(output)

    # create a temporary directory to store the input files for svrtk
    with tempfile.TemporaryDirectory(dir=home) as temp_dir:
        # copy the reconstruction file into the temporary directory
        shutil.copy(input, temp_dir)

        svrtk_input = temp_dir.replace(username,'data')
        svrtk_output = output_directory.replace(username,'data')

        cmd_line = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
        cmd_line+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-reorientation.sh '+svrtk_input+' '+svrtk_output+' 0.5 1 0'
        print(cmd_line)
        os.system(cmd_line)  

    os.rename(os.path.join(output_directory,input_file.replace('.nii.gz','_0_reo.nii.gz')), os.path.join(output_directory,output_file))        

def wrapper_svrtk_segmentation(input, output) -> None:
    home = expanduser("~")
    username = getpass.getuser() # assuming that the home directory is /home/username
    input_file = os.path.basename(input)
    output_directory = os.path.dirname(output)
    output_file = os.path.basename(output)

    #create a temporary folder for bounti
    with tempfile.TemporaryDirectory(dir=home) as temp_dir:
        # copy the reconstruction file into the temporary directory
        shutil.copy(input, temp_dir)

        bounti_input = temp_dir.replace(username,'data')
        bounti_output = output_directory.replace(username,'data')

        cmd_line = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
        cmd_line+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh '+bounti_input+' '+bounti_output
        print(cmd_line)
        os.system(cmd_line)        

    os.rename(os.path.join(output_directory,input_file.replace('.nii.gz','-mask-bet-1.nii.gz')), os.path.join(output_directory,output_file.replace('.nii.gz','-mask.nii.gz')))        
    os.rename(os.path.join(output_directory,input_file.replace('.nii.gz','-mask-brain_bounti-19.nii.gz')), os.path.join(output_directory,output_file.replace('.nii.gz','-bounti.nii.gz')))        
