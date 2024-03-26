#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torchio as tio
from os.path import exists

from unigradicon import make_network

weights_location = "network_weights/unigradicon1.0/Step_2_final.trch"
if not exists(weights_location):
    print("Downloading pretrained model")
    import urllib.request
    import os
    download_path = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
    os.makedirs("network_weights/unigradicon1.0/", exist_ok=True)
    urllib.request.urlretrieve(download_path, weights_location)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo GradICON')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, action='append', required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, action='append', required = True)

    parser.add_argument('-o', '--output', help='Output prefix filename', type=str, required = True)

    args = parser.parse_args()

    source_image = tio.ScalarImage(args.source)
    target_image = tio.ScalarImage(args.target)

    source = torch.unsqueeze(source_image.data,0)
    target = torch.unsqueeze(target_image.data,0)

    input_shape = source.shape

    net = make_network(input_shape, include_last_step=True)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"))
    net.regis_net.load_state_dict(trained_weights)
    net.cuda()
    net.eval()
    with torch.no_grad():
        net(source.cuda(), target.cuda())

    warped_source = net.warped_image_A
    warped_target = net.warped_image_B
    forward_flow = net.phi_AB_vectorfield
    backward_flow = net.phi_BA_vectorfield

    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_image.affine)
    o.save(args.output+'_warped.nii.gz')
    o = tio.ScalarImage(tensor=warped_target[0].detach().numpy(), affine=source_image.affine)   
    o.save(args.output+'_inverse_warped.nii.gz')
    o = tio.ScalarImage(tensor=forward_flow[0].detach().numpy(), affine=target_image.affine)   
    o.save(args.output+'_warp.nii.gz')
    o = tio.ScalarImage(tensor=backward_flow[0].detach().numpy(), affine=source_image.affine)   
    o.save(args.output+'_inverse_warp.nii.gz')
