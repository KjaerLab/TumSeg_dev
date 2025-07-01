#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib.resources as pkg_resources
from typing import Callable, Optional
import torch
import torch.nn as nn
import scipy 
import torchio as tio
import nibabel as nib
import numpy as np
import os 
import joblib

from .unet3d.model import UNet3D
from .tumseg_misc import computeRates, precisionRecallFscore


class TumSeg(nn.Module):
    def __init__(self, net_path, device=None):
        super().__init__()
    
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.net_all = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_all.load_state_dict(torch.load(net_path, weights_only=True))
        
        self.net_all = self.net_all.to(self.device)
        self.net_all.eval()
        
        # load the UQ model
        self.loadUQModel()
        
    def forward(self, X):
        return self.net_all(X)
    
    def init_ensemble(self, net_path_A, net_path_B, net_path_C):
        self.net_A = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_A.load_state_dict(torch.load(net_path_A, weights_only=True))
        self.net_A = self.net_A.to(self.device)
        self.net_A.eval()
        
        self.net_B = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_B.load_state_dict(torch.load(net_path_B, weights_only=True))
        self.net_B = self.net_B.to(self.device)
        self.net_B.eval()

        self.net_C = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_C.load_state_dict(torch.load(net_path_C, weights_only=True))
        self.net_C = self.net_C.to(self.device)
        self.net_C.eval()
    
    def attach_post_processor(self, 
                              post_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                              post_proc_kwargs: Optional[dict] = None):
        self.post_processor = post_processor
        self.post_proc_kwargs = post_proc_kwargs
        
    def attach_post_processor_UQ(self, 
                              post_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                              post_proc_kwargs: Optional[dict] = None):
        
        self.net_all.post_processor = post_processor
        self.net_all.post_proc_kwargs = post_proc_kwargs
        
    def post_process(self, output, CT_in):
        return self.post_processor(output, CT_in, **self.post_proc_kwargs)
    
    def loadUQModel(self):
        self.tumseg_uq_model = joblib.load(pkg_resources.files("TumSeg.networks").joinpath('uq_regression_pipeline.joblib'))
    
    def runUQ(self, subj, n_dropouts = 100, e_p = 0.35, d_p = 0.35):
        
        with torch.no_grad():
            CT_in = subj['CT']['data'].unsqueeze(dim=0)
            do_out = self.net_all.MC_predict(CT_in.to(self.device), n=n_dropouts, 
                                do_p_encoder=e_p, do_p_decoder=d_p, post_process=False, show_progress = True)
            
            
            entropy_masked = self.net_all.MC_entropy(do_out, CT_in = CT_in, reduction='mean', method='entropy',
                                      masking = True)
        
            cross_entropy_binary_masked = self.net_all.MC_entropy(do_out,CT_in = CT_in, reduction='mean', method='cross-entropy-binary',
                                      masking = True)
            
            fleiss_kappa = self.net_all.MC_fleissKappa(do_out, CT_in=CT_in, post_process = True, chunks = 10)
            
            # Run the ensemble 
            output_A = self.net_A(CT_in.to(self.device))
            output_A = output_A.softmax(dim=1)
            output_A = output_A[0,1,:,:,:].detach().cpu()
            output_A = self.post_process(output_A, CT_in[0,0])            
                        
            output_B = self.net_A(CT_in.to(self.device))
            output_B = output_B.softmax(dim=1)
            output_B = output_B[0,1,:,:,:].detach().cpu()
            output_B = self.post_process(output_B, CT_in[0,0])                    
            
            output_C = self.net_A(CT_in.to(self.device))
            output_C = output_C.softmax(dim=1)
            output_C = output_C[0,1,:,:,:].detach().cpu()
            output_C = self.post_process(output_C, CT_in[0,0])            
            
            dice_ensemble = np.mean(getAllF1Combinations(output_A,output_B,output_C))
            
        
        X_uq = np.transpose(np.vstack([fleiss_kappa, entropy_masked, cross_entropy_binary_masked, dice_ensemble]))
        
        return self.tumseg_uq_model.predict(X_uq)[0]
    
    

def getAllF1Combinations(roi_a, roi_b, roi_c, reference=None):
    tmp_f1 = []
    
    if reference is None:
        TP, TN, FP, FN = computeRates(roi_a, roi_b, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_a, roi_c, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_b, roi_c, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
    else:
        TP, TN, FP, FN = computeRates(roi_a, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_b, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_c, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
    
    return tmp_f1


def windowCT(min_=-400, max_=400):
    '''
    Helper function for the transform pipeline
    '''    
    def windowCT_(subject):
        subject['CT']['data'][subject['CT']['data'] < min_] = min_ 
        subject['CT']['data'][subject['CT']['data'] > max_] = max_ 
        
        return subject
    
    return windowCT_

def nifti_loader_float(path):
    img_prox = nib.load(path)
    # return torch.Tensor(img_prox.get_fdata()).type(torch.FloatTensor), np.eye(4)
    return torch.tensor(img_prox.get_fdata(), dtype=torch.float32), img_prox.header.get_base_affine()

def buildSubjectList(input_path):
    subjects = []
    # input is a single nifti file
    if input_path.endswith('nii') or input_path.endswith('nii.gz'):
        ct_path = input_path
        subjects.append(tio.Subject(
            CT = tio.ScalarImage(ct_path, reader=nifti_loader_float),
            CT_prox = nib.load(ct_path),
            path = ct_path
        ))
    # input is a folder
    else:
        for path, folders, files in os.walk(input_path):
            if files:
                for file in files:
                    if file.endswith('nii') or file.endswith('nii.gz'):
                        ct_path = os.path.join(path,file)
                        subjects.append(tio.Subject(
                            CT = tio.ScalarImage(ct_path, reader=nifti_loader_float),
                            CT_prox = nib.load(ct_path),
                            path = ct_path
                        ))
        
    return subjects

def runInference(subj, tumseg, patch_size=(128, 128, 128)):
    floats_per_gb = (208**3) / 11.94 # Assuming 128x128x128 is the max patch size for 12GB of VRAM
    freemem, totalmem = torch.cuda.mem_get_info()
    num_floats = (totalmem/2**30) * floats_per_gb # Very rough estimate
    scan_shape = subj.shape[-3:]
    print(f'Predicting scan of shape {scan_shape}')

    if scan_shape[0]*scan_shape[1]*scan_shape[2] < num_floats:
        # Run inference on the whole scan
        print(f'Model should be able to run prediction on the whole scan {scan_shape}, trying out...')
        try: 
            tumseg.eval()
            with torch.no_grad():
                CT_in = subj['CT']['data'].unsqueeze(dim=0)
                output = tumseg(CT_in.to(tumseg.device))
                output = output.softmax(dim=1)
                
                output = output[0,1,:,:,:].detach().cpu()
            return output
        except Exception as e:
            print(f'Prediction on the whole scan failed with error: {e}. Trying again with patches...')
    
    print('Running inference on patches...')
    # Run inference on patches
    # Patch size can't be bigger than the smallest axis
    sorted_shape = sorted(scan_shape)
    patch_side = int(num_floats**(1/3))
    print(f'Initial patch size based on free memory: ({patch_side}x{patch_side}x{patch_side})')
    # If the patch side is larger than the smallest axis, increase it fore the other 2
    if sorted_shape[0] < patch_side:
        patch_side = int((num_floats/(sorted_shape[0]))**(1/2))
    # If the patch side is now larger than the second smallest axis, increase it for the last one
    if sorted_shape[1] < patch_side:
        patch_side = int(num_floats/(sorted_shape[0] * sorted_shape[1]))
    # The patch side can't be larger than all the axes, as we would have predicted the whole scan    
    max_patch_size = [min(axis_size, patch_side) for axis_size in scan_shape]

    print(f'Max patch size based on free memory: {max_patch_size}')

    sampler = tio.inference.GridSampler(subj, max_patch_size, patch_overlap=[s//8*2 for s in max_patch_size])
    aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')
    tumseg.eval()
    with torch.no_grad():
        for idx, patch in enumerate(sampler):
            print(f'Processing patch {idx}/{len(sampler)}')
            patch_data = patch['CT']['data'].unsqueeze(0).to(tumseg.device)
            output = tumseg(patch_data)
            output = output.softmax(dim=1).detach().cpu()

            # Output is of shape: (Batches, Classes, D, H, W), location shape: (Batches, 6)
            aggregator.add_batch(output, patch[tio.LOCATION].unsqueeze(0))
        
    output = aggregator.get_output_tensor()
    output = output[1,:,:,:]
    return output

def resampleAndPostProcess(output, subj,tumseg, target_pixel_size):
    # inverse transform and post process 
    target_inverse_zoom = np.array(target_pixel_size) / np.array(subj['CT_prox'].header.get_zooms())
    output = scipy.ndimage.zoom(output, zoom=target_inverse_zoom, order=1) > 0.5 
    CT_in = nib.load(subj['path']).get_fdata()
    CT_in = np.clip(CT_in, -400, 400)
    output = tumseg.post_process(output, CT_in)
    
    return output

def saveResults(output, subj, input_path, output_path, uq_pred=None):
    # Save results as a nifti 
    rel_path = os.path.relpath('/'.join(subj['path'].split(os.sep)[:-1]), input_path)
    os.makedirs(os.path.join(output_path, rel_path), exist_ok=True)
    new_nii = nib.Nifti1Image(output.astype(np.int8), subj['CT_prox'].affine, subj['CT_prox'].header)
    scan_name = subj['path'].split(os.sep)[-1].split('.')[0]
    full_save_path = os.path.join(output_path, rel_path, scan_name + '_tumor_mask.nii.gz')
    print('Saving results to: ' + full_save_path)
    nib.save(new_nii, full_save_path)
    
    if uq_pred is not None:
        with open(os.path.join(output_path, rel_path, 'uq_dice_prediction.txt'), 'w') as f:
            f.write('{:.3f}'.format(uq_pred))
         

#------------------------------------------------------------------------------------------------------------------------------
# LOADING AND PREDICTION FUNCTIONS FOR ARRAYS PAST THIS LINE, TREAD CAREFULLY (NOT REALLY)
#------------------------------------------------------------------------------------------------------------------------------

def buildSubjectListArrays(arrays, affines):
    subjects = []
    for arr, aff in zip(arrays, affines):
        # The input to scalar image has to be 4D (CxWxHxD)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        subjects.append(tio.Subject(
            CT = tio.ScalarImage(tensor=torch.tensor(arr, dtype=torch.float32), affine=aff),
            voxel_size = np.abs(np.diagonal(aff)[:3]),
            shape = arr.shape[1:]
        ))

    return subjects

def resampleAndPostProcessArray(output, subj, tumseg):
    # inverse transform and post process 
    # using pixel size sometimes results in incorrect output mask size
    target_inverse_zoom = np.array(subj['shape']) / np.array(subj['CT']['data'].shape[1:])
    # Threshold at 0.5
    output = scipy.ndimage.zoom(output, zoom=target_inverse_zoom, order=1) > 0.5 
    CT_in = np.squeeze(subj['CT']['data'].numpy())
    CT_in = np.clip(CT_in, -400, 400)
    output = tumseg.post_process(output, CT_in)
    
    return output
