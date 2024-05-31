#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
home = expanduser("~")
import glob
import argparse
import tempfile
import shutil
import getpass
import nibabel as nib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Fetal Reconstruction')
    parser.add_argument('-i', '--input', help='Input folder (absolute path)', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output folder (absolute path)', type=str, required=True)
    parser.add_argument('-k', '--keyword', help='Keyword used to select images (like HASTE ou TrueFISP)', type=str, required=True)
    parser.add_argument('-m', '--masking', help='Masking method (nesvor or niftymic)', type=str, required=False, default='nesvor')
    parser.add_argument('-r', '--recon', help='Reconstruction method (nesvor, niftymic, svrtk, all)', type=str, required=False, default='nesvor')

    args = parser.parse_args()
    username = getpass.getuser() # assuming that the home directory is /home/username

    #Find automatically all images in input directory (available from the home directory)
    raw_stacks = []
    files = glob.glob(os.path.join(args.input,'*.nii.gz'))
    for f in files:
        if args.keyword in f:
            raw_stacks.append(f)
    print('List of input raw stacks:')    
    print(raw_stacks)      

    print('Denoising using SCUNet')
    denoised_stacks = []
    for file in raw_stacks:
        # lines to modify according to the path of the SCUNet model
        cmd_line = 'time python '+os.path.join(home, 'Sync-Exp','SCUNet','main_3dmri.py')
        cmd_line+= ' --model_zoo '+os.path.join(home, 'Sync-Exp','SCUNet','model_zoo')
        cmd_line+= ' -i '+file
        outputfile = os.path.join(args.output, os.path.basename(file).replace('.nii.gz','_denoised.nii.gz'))
        denoised_stacks.append(outputfile)
        cmd_line+= ' -o '+outputfile
        print(cmd_line)
        os.system(cmd_line)

    if args.masking == 'nesvor':
        print('Brain masking using Nesvor')
        #Dell IMT (issue svd with nibabel)
        mask_stacks = []
        cmd_line = 'time docker run --rm --gpus all --ipc=host '
        cmd_line += '-v ' + args.input + ':/incoming:ro '
        cmd_line += '-v ' + args.output + ':/outgoing:rw '
        cmd_line += 'junshenxu/nesvor '
        cmd_line += 'nesvor segment-stack '
        cmd_line += '--input-stacks '
        for file in denoised_stacks:            
            cmd_line += '/outgoing/' + os.path.basename(file) + ' '
        cmd_line += '--output-stack-masks '
        for file in denoised_stacks:
            outputfile = os.path.basename(file).replace('.nii.gz','_mask.nii.gz')
            mask_stacks.append(os.path.join(args.output, outputfile))                    
            cmd_line += '/outgoing/' + outputfile + ' '

        print(cmd_line)
        os.system(cmd_line)

    else:    
        print('Brain masking using niftyMIC')
        cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_segment_fetal_brains '
        docker_stacks = [s.replace(username,'data') for s in denoised_stacks]
        docker_masks  = [s.replace('.nii.gz','_mask.nii.gz') for s in docker_stacks]

        cmd_os+= ' --filenames '
        for i in docker_stacks:
            cmd_os+= i+' '
        cmd_os+= ' --filenames-masks '
        for i in docker_masks:
            cmd_os+= i+' '

        print(cmd_os)
        os.system(cmd_os)

    mask_stacks = [s.replace('.nii.gz','_mask.nii.gz') for s in denoised_stacks]
    prefix = os.path.commonprefix([os.path.basename(s) for s in denoised_stacks])
    if args.recon == 'nesvor' or args.recon == 'all':
        print('Reconstruction using nesvor')
        cmd_os = 'time docker run --rm --gpus all --ipc=host '
        cmd_os += '-v ' + args.input + ':/incoming:ro '
        cmd_os += '-v ' + args.output + ':/outgoing:rw '
        cmd_os += 'junshenxu/nesvor '
        cmd_os += 'nesvor reconstruct '
        cmd_os += '--input-stacks '
        for i in denoised_stacks:            
            cmd_os += '/outgoing/' + os.path.basename(i) + ' '
        cmd_os+= '--stack-masks '
        for i in mask_stacks:                    
            cmd_os += '/outgoing/' + os.path.basename(i) + ' '

        cmd_os += ' --bias-field-correction --output-resolution 6 ' 
        cmd_os += '--output-volume '
        cmd_os += '/outgoing/'+prefix+'nesvor_r6.nii.gz '
        cmd_os += '--output-model /outgoing/'+prefix+'nesvor.pt '

        print(cmd_os)
        os.system(cmd_os)

        cmd_os = 'time docker run --rm --gpus all --ipc=host '
        cmd_os += '-v ' + args.input + ':/incoming:ro '
        cmd_os += '-v ' + args.output + ':/outgoing:rw '
        cmd_os += 'junshenxu/nesvor '
        cmd_os += 'nesvor sample-volume --inference-batch-size 2048 --verbose 2 --output-volume '
        cmd_os += '/outgoing/'+prefix+'nesvor_r05.nii.gz '
        cmd_os += ' --output-resolution 0.5 '
        cmd_os += '--input-model /outgoing/'+prefix+'nesvor.pt '

        print(cmd_os)
        os.system(cmd_os)        
        os.remove(os.path.join(args.output,prefix+'nesvor_r6.nii.gz'))

    if args.recon == 'niftymic' or args.recon == 'all':
        print('Reconstruction using niftymic')
        cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_reconstruct_volume '
        docker_output = args.output
        docker_output = docker_output.replace(username,'data')        
        cmd_os+= ' --output '+os.path.join(docker_output,prefix+'niftymic_r05.nii.gz')+' --isotropic-resolution 0.5 '

        cmd_os+= ' --filenames '
        docker_stacks = [s.replace(username,'data') for s in denoised_stacks]
        docker_masks = [s.replace(username,'data') for s in mask_stacks]

        for i in docker_stacks:
            cmd_os+= i+' '
        cmd_os+= ' --filenames-masks '
        for i in docker_masks:
            cmd_os+= i+' '

        print(cmd_os)
        os.system(cmd_os)    

    if args.recon == 'svrtk' or args.recon == 'all':
        print('Reconstruction using SVRTK')

        # Create a temporary directory to store the input files for svrtk
        with tempfile.TemporaryDirectory(dir=home) as temp_dir:
            # temp_dir is the path to the temporary directory
            print(f'Temporary directory: {temp_dir}')

            # Copy files into the temporary directory
            for f in denoised_stacks:
                shutil.copy(f, temp_dir)

            svrtk_input = temp_dir.replace(username,'data')
            svrtk_output = args.output.replace(username,'data')

            slice_thickness = max(nib.load(denoised_stacks[0]).header['pixdim'])
            print('Slice thickness:',slice_thickness)

            cmd_os = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
            cmd_os+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-reconstruction.sh '+svrtk_input+' '+svrtk_output+' 1 '+str(slice_thickness)+' 0.5 1'
            print(cmd_os)
            os.system(cmd_os)  
            os.rename(os.path.join(args.output,'reo-SVR-output-brain.nii.gz'), os.path.join(args.output,prefix+'svrtk_r05.nii.gz'))  

    # reorient the reconstructed images using svrtk docker
    recon_method = ['nesvor','niftymic','svrtk']
    for r in recon_method:
        recon_file = os.path.join(args.output,prefix+r+'_r05.nii.gz')
        if os.path.exists(recon_file):
            print('Reconstruction file to be reoriented:',recon_file)

            # create a temporary directory to store the input files for svrtk
            with tempfile.TemporaryDirectory(dir=home) as temp_dir:
                # copy the reconstruction file into the temporary directory
                shutil.copy(recon_file, temp_dir)

                svrtk_input = temp_dir.replace(username,'data')
                svrtk_output = args.output.replace(username,'data')

                cmd_os = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
                cmd_os+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-reorientation.sh '+svrtk_input+' '+svrtk_output+' 0.5 1 0'
                print(cmd_os)
                os.system(cmd_os)


    # Segmentation using bounti using svrtk docker
    recon_method = ['nesvor','niftymic','svrtk']
    for r in recon_method:
        recon_file = os.path.join(args.output,prefix+r+'_r05_0_reo.nii.gz')
        if os.path.exists(recon_file):
            print('Reconstruction file to be segmented:',recon_file)
            
            #create a temporary folder for bounti
            with tempfile.TemporaryDirectory(dir=home) as temp_dir:
                # copy the reconstruction file into the temporary directory
                shutil.copy(recon_file, temp_dir)

                bounti_input = temp_dir.replace(username,'data')
                bounti_output = args.output.replace(username,'data')

                cmd_os = 'time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
                cmd_os+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh '+bounti_input+' '+bounti_output
                print(cmd_os)
                os.system(cmd_os)
