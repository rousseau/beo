#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
import getpass
import shutil

def wrapper_scunet(input, output, noise=None):
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

def wrapper_masking(input, output, method):
    if method == 'nesvor':
        wrapper_nesvor_masking(input, output)
    elif method == 'niftymic':
        wrapper_niftymic_masking(input, output)
    elif method == 'synthstrip':
        wrapper_synthstrip_masking(input, output)
    else:
        print('Method not implemented')
        return

def wrapper_nesvor_masking(input, output):
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

def wrapper_niftymic_masking(input,output):
    print('Brain masking using niftyMIC')
    username = getpass.getuser() # assuming that the home directory is /home/username
    home = expanduser("~")
    cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_segment_fetal_brains '
    cmd_os+= ' --filenames '+input.replace(username,'data')+' --filenames-masks '+output.replace(username,'data')
    print(cmd_os)
    os.system(cmd_os)    

def wrapper_synthstrip_masking(input,output):
    print('Brain masking using mri_synthstrip')
    cmd_os = 'mri_synthstrip -i '+input+' -m '+output
    print(cmd_os)
    os.system(cmd_os)