import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import SimpleITK

import json
import time
import tempfile
import hashlib

import cell_locator_utils

def do_analysis():

    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)
    print(img_data.shape)
    t0 = time.time()
    nx0 = img_data.shape[2]
    ny0 = img_data.shape[1]
    nz0 = img_data.shape[0]
    img_data = img_data.flatten()

    allen_coords = np.zeros((3,nx0*ny0*nz0), dtype=float)

    mesh = np.meshgrid(resolution*np.arange(nx0),
                       resolution*np.arange(ny0),
                       resolution*np.arange(nz0),
                       indexing = 'ij')

    allen_coords[2,:] = mesh.pop(2).flatten()
    allen_coords[1,:] = mesh.pop(1).flatten()
    allen_coords[0,:] = mesh.pop(0).flatten()

    print('grid created in %e seconds\n' % (time.time()-t0))

    for dex in range(4):
        t1 = time.time()

        annotation_name = '../CellLocatorAnnotations/annotation_%d.json' % dex
        img_name = tempfile.mkstemp(dir='.',
                                prefix='annotation_%d_' % dex,
                                suffix='.png')[1]
        control_name = 'control_imgs/annotation_%d.png' % dex
        assert os.path.isfile(control_name)

        with open(annotation_name, 'rb') as in_file:
            annotation = json.load(in_file)

        coord_converter = cell_locator_utils.CellLocatorTransformation(annotation)

        valid_dex = np.where(coord_converter.get_slice_mask_from_allen(allen_coords,
                                                                       resolution))
        print('valid_dex after %e seconds' % (time.time()-t1))

        slice_coords = coord_converter.allen_to_slice(allen_coords[:,valid_dex[0]])

        img_x_min = slice_coords[0,:].min()
        img_x_max = slice_coords[0,:].max()
        img_y_min = slice_coords[1,:].min()
        img_y_max = slice_coords[1,:].max()

        img_x_min = np.round(img_x_min/resolution).astype(int)
        img_x_max = np.round(img_x_max/resolution).astype(int)
        img_y_min = np.round(img_y_min/resolution).astype(int)
        img_y_max = np.round(img_y_max/resolution).astype(int)

        n_img_x = img_x_max-img_x_min+1
        n_img_y = img_y_max-img_y_min+1

        new_img_pts = np.zeros((2, n_img_x*n_img_y), dtype=float)

        mesh = np.meshgrid(resolution*(img_x_min+np.arange(n_img_x)),
                           resolution*(img_y_min+np.arange(n_img_y)),
                           indexing='ij')

        new_img_pts[1,:] = mesh.pop(1).flatten()
        new_img_pts[0,:] = mesh.pop(0).flatten()

        print('dummy image after %e seconds' % (time.time()-t1))

        new_allen_coords = coord_converter.slice_to_allen(new_img_pts)
        new_allen_dexes = np.round(new_allen_coords/resolution).astype(int)
        print('allen dexes after %e seconds' % (time.time()-t1))

        valid_dex = np.where(np.logical_and(new_allen_dexes[0,:]>=0,
                         np.logical_and(new_allen_dexes[0,:]<nx0,
                         np.logical_and(new_allen_dexes[1,:]>=0,
                         np.logical_and(new_allen_dexes[1,:]<ny0,
                         np.logical_and(new_allen_dexes[2,:]>=0,
                                        new_allen_dexes[2,:]<nz0))))))

        img_ix = (new_img_pts[0,:]/resolution-img_x_min).astype(int)
        img_iy = (new_img_pts[1,:]/resolution-img_y_min).astype(int)

        new_img = np.zeros(n_img_x*n_img_y, dtype=float)
        ix_arr = img_ix[valid_dex]
        iy_arr = n_img_y-1-img_iy[valid_dex]
        ii_flat = ix_arr*n_img_y+iy_arr

        ax_arr = new_allen_dexes[0,valid_dex]
        ay_arr = new_allen_dexes[1,valid_dex]
        az_arr = new_allen_dexes[2,valid_dex]


        img_dex_flat = az_arr*(nx0*ny0)+ay_arr*nx0+ax_arr
        pixel_vals = img_data[img_dex_flat]
        new_img[ii_flat] = pixel_vals
        new_img = new_img.reshape(n_img_x, n_img_y)

        new_img = new_img.transpose()
        print('got img data in %e seconds' % (time.time()-t1))

        plt.figure(figsize=(10,10))
        plt.imshow(new_img)
        plt.savefig(img_name)
        print(img_name)

        md5_control = hashlib.md5()
        with open(control_name, 'rb') as in_file:
            while True:
                data = in_file.read(10000)
                if len(data)==0:
                    break
                md5_control.update(data)
        md5_test = hashlib.md5()
        with open(img_name, 'rb') as in_file:
            while True:
                data = in_file.read(10000)
                if len(data)==0:
                    break
                md5_test.update(data)
        assert md5_test.hexdigest() == md5_control.hexdigest()

        print(md5_test.hexdigest())
        print(md5_control.hexdigest())
        print('')
        if os.path.exists(img_name):
            os.unlink(img_name)

if __name__ == "__main__":
    do_analysis()
