import argparse
import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from DeepC_3D_Patch_test_model.TABS_Model import TABS
from numpy.typing import NDArray
import time

parser = argparse.ArgumentParser()

parser.add_argument('--inp', type=str)

parser.add_argument('--output', type=str)

parser.add_argument('--model_path', default='./checkpoint/TABS-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl', type=str)

parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

def center_crop(img, dim, axis):
    shape = img.shape
    start = shape[axis]//2 - dim//2

    if axis == 0:
    	return img[start:start+dim, :, :], start, shape[axis] - (start+dim)
    if axis == 1:
    	return img[:, start:start+dim, :], start, shape[axis] - (start+dim)
    if axis == 2:
    	return img[:, :, start:start+dim], start, shape[axis] - (start+dim)

def crop_back(img, bottom_crop, top_crop, axis):
    shape = img.shape
    if axis == 0:
    	return img[bottom_crop:(shape[0]-top_crop), :, :]
    if axis == 1:
    	return img[:, bottom_crop:(shape[1]-top_crop), :]
    if axis == 2:
    	return img[:, :, bottom_crop:(shape[2]-top_crop)]

def pad(img, dim, axis):
    shape = img.shape
    to_increase = shape[axis]
    diff = dim - to_increase
    diff_div = diff // 2
    top_add = diff - diff_div

    if axis == 0:
    	return np.pad(img, [(diff_div, top_add), (0, 0), (0,0)], mode='constant'), diff_div, top_add
    if axis == 1:
    	return np.pad(img, [(0, 0), (diff_div, top_add), (0,0)], mode='constant'), diff_div, top_add
    if axis == 2:
    	return np.pad(img, [(0, 0), (0,0), (diff_div, top_add)], mode='constant'), diff_div, top_add

def pad_back(img, bottom_pad, top_pad, axis):
    if axis == 0:
    	return np.pad(img, [(bottom_pad, top_pad), (0, 0), (0,0)], mode='constant')
    if axis == 1:
    	return np.pad(img, [(0, 0), (bottom_pad, top_pad), (0,0)], mode='constant')
    if axis == 2:
    	return np.pad(img, [(0, 0), (0,0), (bottom_pad, top_pad)], mode='constant')

def sequential_patch(image: NDArray, patch_size, step):

    (H, W, D) = image.shape  # (144,208,208)
    patch_total_num = (1+len(range(0, D - patch_size, step))) * (1+len(range(0, W - patch_size, step))) * (1+len(range(0, H - patch_size, step)))
    count=0
    coordinate_list = []
    patch_mat = np.float32(np.zeros((patch_total_num,patch_size,patch_size,patch_size)))
    #image_zeropadding = patch/2
    for z in range(0, D - patch_size, step):
        for y in range(0, W- patch_size, step):
            for x in range(0, H - patch_size, step):

                patch = image[x : x + patch_size, y : y + patch_size, z : z + patch_size]
                patch_mat[count,:,:,:]=patch

                coordinate = (x, y, z)
                coordinate_list.append(coordinate)
                count=count+1
                del patch

    return patch_mat, coordinate_list, patch_total_num

def reconstruction(patch_mat,coordinate_list,image_shape,patch_size):
    (H, W, D) = image_shape
    map = np.zeros((H, W, D))
    repeat = np.zeros((H, W, D))
    patch_mat[np.isnan(patch_mat)] = 0

    for index in range(len(coordinate_list)):
        prep = np.squeeze(patch_mat[index,:,:,:])
        (x,y,z)=coordinate_list[index]
        map[x : x + patch_size, y : y + patch_size, z : z + patch_size] = map[x : x + patch_size, y : y + patch_size, z : z + patch_size] + prep
        repeat[x : x + patch_size, y : y + patch_size, z : z + patch_size] = repeat[x : x + patch_size, y : y + patch_size, z : z + patch_size] + 1
    map = map/repeat
    return map

def zero_pad(scan):
    post_padding_dimension = 240
    pad_val = int((post_padding_dimension - scan.shape[1])/2)
    padded = np.pad(scan,((pad_val,pad_val),(pad_val,pad_val), (0,0)), mode='constant', constant_values=0)
    padded = padded[:,:,0:155]

    return padded

if __name__ == '__main__':
    start_time = time.time()

    unet = TABS(img_dim=96)
    unet.load_state_dict(torch.load(args.model_path))
    print('loaded best model')
    unet = unet.cuda(args.device)
    unet.train(False)
    unet.eval()

    files = sorted(glob(args.inp + '*.nii.gz'))
    ids = [i.split('.nii.gz')[0] for i in files]
    ids = [i.split('/')[-1] for i in ids]

    print('performing inference on {} files'.format(len(files)))

    for j in range(0,len(files)):

        pre_path = files[j]

        pre_nifti_scan = nib.load(pre_path)
        pre_nifti_scan = np.array(pre_nifti_scan.dataobj)

        # turn original scan to original scan into 192x192x160
        #####################

        x, y, z = pre_nifti_scan.shape
        x_crop, y_crop, z_crop, x_pad, y_pad, z_pad = False, False, False, False, False, False

        if x > 192:
        	pre_nifti_scan, bottom_pad_x, top_pad_x  = center_crop(pre_nifti_scan,192,0)
        	x_crop = True
        if x < 192:
        	pre_nifti_scan, bottom_crop_x, top_crop_x  = pad(pre_nifti_scan,192,0)
        	x_pad = True

        if y > 192:
        	y_crop = True
        	pre_nifti_scan, bottom_pad_y, top_pad_y  = center_crop(pre_nifti_scan,192,1)
        if y < 192:
        	y_pad = True
        	pre_nifti_scan, bottom_crop_y, top_crop_y  = pad(pre_nifti_scan,192,1)

        if z > 160:
        	z_crop = True
        	pre_nifti_scan, bottom_pad_z, top_pad_z  = center_crop(pre_nifti_scan,160,2)
        if z < 160:
        	z_pad = True
        	pre_nifti_scan, bottom_crop_z, top_crop_z  = pad(pre_nifti_scan,160,2)

        #####################

        pre_nifti_scan = pre_nifti_scan.transpose((-1,0,1))
        pre_nifti_scan = torch.from_numpy(pre_nifti_scan)
        brain_mask = (pre_nifti_scan > 0).int()
        brain_mask = brain_mask.numpy()

        patch_size=96
        pre_image_norms, coordinate_list, _ =  sequential_patch(pre_nifti_scan.numpy(), patch_size, step=6)
        coordinate_list = np.array(coordinate_list)
        pre_image_norms = torch.from_numpy(pre_image_norms)
        pre_image_norms = pre_image_norms.unsqueeze(0)

        with torch.no_grad():
            patch_total_num = pre_image_norms.shape[1]
            prediction_mat=np.float32(np.zeros((patch_total_num,96,96,96)))
            for i in range(0,patch_total_num):
                pre_image_norm = pre_image_norms[:,i]
                inp_image = pre_image_norm.to(args.device)
                # Create a 5th dimension, as needed by 3D convolution
                inp_image = inp_image[:,np.newaxis,:,:,:]
                # Use the network to make predictions with pre_image as input.
                cur_prediction = unet(inp_image)[0]
                cur_prediction = cur_prediction.detach().cpu().numpy()
                prediction_mat[i,:,:,:] = cur_prediction[0]
                del cur_prediction

            Prediction=reconstruction(prediction_mat,coordinate_list,(160,192,192),96)
            Prediction[np.isnan(Prediction)] = 0

        Prediction = Prediction * brain_mask

        corresponding_pre_nifti = nib.load(pre_path)
        current_scan_affine = corresponding_pre_nifti.affine
        current_scan_header = corresponding_pre_nifti.header

        # back to original dimensions
        ####################
        Prediction = Prediction.transpose((1,-1,0))
        brain_mask = Prediction.transpose((1,-1,0))

        if x_crop == True:
        	Prediction  = pad_back(Prediction,bottom_pad_x,top_pad_x,0)
        if x_pad == True:
        	Prediction  = crop_back(Prediction,bottom_crop_x, top_crop_x,0)
        if y_crop == True:
        	Prediction  = pad_back(Prediction,bottom_pad_y,top_pad_y,1)
        if y_pad == True:
        	Prediction  = crop_back(Prediction,bottom_crop_y, top_crop_y,1)
        if z_crop == True:
        	Prediction  = pad_back(Prediction,bottom_pad_z,top_pad_z,2)
        if z_pad == True:
        	Prediction  = crop_back(Prediction,bottom_crop_z, top_crop_z,2)

        ####################
        current_prediction_scan_nifti = nib.Nifti1Image(Prediction, current_scan_affine, current_scan_header)
        print(args.output + ids[j] + '_cbv.nii.gz')
        nib.save(current_prediction_scan_nifti, args.output + ids[j] + '_cbv.nii.gz')

        del pre_image_norms, coordinate_list, pre_nifti_scan, brain_mask, pre_image_norm, inp_image, prediction_mat, Prediction

    print('time taken: {}'.format(time.time() - start_time))
