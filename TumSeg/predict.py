#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib.resources as pkg_resources
import torchio as tio
import numpy as np
import argparse

from .tumseg_misc import postProcessROIs
from .modules import TumSeg, buildSubjectList, runInference, resampleAndPostProcess, saveResults, windowCT, buildSubjectListArrays, resampleAndPostProcessArray
from torch import cuda

def setup_model(device=None):
    ''' Setup TumSeg '''
    net_path_A = pkg_resources.files("TumSeg.networks").joinpath('network_annotator_A.pt') #'./networks/network_annotator_A.pt'
    net_path_B = pkg_resources.files("TumSeg.networks").joinpath('network_annotator_B.pt') #'./networks/network_annotator_B.pt'
    net_path_C = pkg_resources.files("TumSeg.networks").joinpath('network_annotator_C.pt') #'./networks/network_annotator_C.pt'
    net_path_all = pkg_resources.files("TumSeg.networks").joinpath('network_annotator_A-C.pt') #'./networks/network_annotator_A-C.pt'
    
    if device is None:
        device = 'cuda' if cuda.is_available() else 'cpu'
    tumseg = TumSeg(net_path_all, device = device)
    # initialize the ensemble 
    tumseg.init_ensemble(net_path_A, net_path_B, net_path_C)
    
    # Attach the post processing module 
    
    post_proc_kwargs = {
            'classify_thres': 0.5, 
            'size_thres': 0.2, 
            'remove_intensity': False, 
            'verbose': False
            }
    
    post_proc_kwargs_UQ = {
            'classify_thres': 0.5, 
            'size_thres': 0.1, 
            'remove_intensity': True, 
            'verbose': False,
            'intensity_thres': 0.99, 
            'intensity_perc_thres': 0.1
            }
    
    tumseg.attach_post_processor(post_processor = postProcessROIs, 
                                 post_proc_kwargs = post_proc_kwargs)
    
    tumseg.attach_post_processor_UQ(post_processor = postProcessROIs, 
                                 post_proc_kwargs = post_proc_kwargs_UQ)
    
    return tumseg

def pred_arrays(arrays, affines, permute=(1,0,2), run_uq=False):
    # Axis order has to be yxz after permutation. 
    # The default permutation (1, 0, 2) is defined for xyz arrays. and (1,2,0) would be for zyx arrays
    tumseg = setup_model()

    '''Setup data to run '''
    target_pixel_size = (0.420, 0.420, 0.420)

    transform = tio.Compose([
        windowCT(-400, 400),
        tio.RescaleIntensity([0,1]),
        tio.Resample(target_pixel_size) 
    ])

    if permute is not None:
        arrays = [np.transpose(arr, permute) for arr in arrays]
        # Reflecting permutation on the affines
        for aff in affines:
            ordering = list(permute)+[3]
            np.fill_diagonal(aff, np.diagonal(aff)[ordering])
            aff[:,-1] = aff[:,-1][ordering]

    subjects = buildSubjectListArrays(arrays, affines)
    print(f'Found {len(subjects)} scans')

    dataloader = tio.SubjectsDataset(subjects, transform=transform)

    output = []
    for idx, subj in enumerate(dataloader):
        print(f'Analyzing scan {idx}')
        out = runInference(subj, tumseg)
        
        print('resampling..')
        out = resampleAndPostProcessArray(out, subj, tumseg)
        if permute is not None:
            inv_permute = np.argsort(permute)
            out = np.transpose(out, inv_permute)
        output.append(out)
        
        if run_uq:
            print('Running Monte-Carlo samples for UQ..')
            uq_pred = tumseg.runUQ(subj)
            if uq_pred < 0:
                uq_pred = 0
            elif uq_pred > 1:
                uq_pred = 1
                
            print('')
            print(f'Expected Dice score: {uq_pred:.3f}')
            print('')

        print('Done.\n\n')

    return output
    

def main(input_path, output_path, device=None, run_uq=False):
    tumseg = setup_model(args.device)
    
    '''Setup data to run '''
    target_pixel_size = (0.420, 0.420, 0.420)
    
    transform = tio.Compose([
        windowCT(-400, 400),
        tio.RescaleIntensity([0,1]),
        tio.Resample(target_pixel_size) 
    ])
    
    subjects = buildSubjectList(args.input_path)
    print(f'Found {len(subjects)} scans')
    
    dataloader = tio.SubjectsDataset(subjects, transform=transform)
    
    for subj in dataloader:
        print('Analyzing: ' + subj['path'])
        output = runInference(subj, tumseg)
        
        print('resampling..')
        output = resampleAndPostProcess(output, subj, tumseg, target_pixel_size)
        
        if args.run_uq:
            print('Running Monte-Carlo samples for UQ..')
            uq_pred = tumseg.runUQ(subj)
            if uq_pred < 0:
                uq_pred = 0
            elif uq_pred > 1:
                uq_pred = 1
                
            print('')
            print('Expected Dice score: {:.3f}'.format(uq_pred))
            print('')
            
            saveResults(output, subj, args.input_path, args.output_path, uq_pred=uq_pred)
        else:
            saveResults(output, subj, args.input_path, args.output_path, uq_pred=None)

        print('Done.\n\n')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help="Path to the input must be either a folder containing nifti files or ne a single file ending with .nii or .nii.gz")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument('--device', type=str, required=False,
                        help="Device to run prediction on (cuda, cpu)")
    parser.add_argument('--run_uq', action='store_true', default=False,
                        help="Run UQ (default: False)")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.device, args.run_uq)















