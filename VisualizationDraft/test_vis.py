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

    allen_coords = np.zeros((3,nx0*ny0*nz0), dtype=float)

    mesh = np.meshgrid(resolution*np.arange(nx0),
                       resolution*np.arange(ny0),
                       resolution*np.arange(nz0),
                       indexing = 'ij')

    allen_coords[2,:] = mesh.pop(2).flatten()
    allen_coords[1,:] = mesh.pop(1).flatten()
    allen_coords[0,:] = mesh.pop(0).flatten()

    print('grid created in %e seconds' % (time.time()-t0))

    for dex in range(4):
        annotation_name = '../CellLocatorAnnotations/annotation_%d.json' % dex
        img_name = tempfile.mkstemp(dir='.',
                                prefix='annotation_%d_' % dex,
                                suffix='.png')[1]
        control_name = 'control_imgs/annotation_%d.png' % dex
        assert os.path.isfile(control_name)

        with open(annotation_name, 'rb') as in_file:
            annotation = json.load(in_file)

        coord_converter = cell_locator_utils.CellLocatorTransformation(annotation)

        t1 = time.time()
        slice_coords = coord_converter.allen_to_slice(allen_coords)
        print('slice_coords after %e seconds' % (time.time()-t1))

        valid_dex = np.where(np.abs(slice_coords[2,:])<0.5*resolution)

        img_x_min = slice_coords[0,valid_dex].min()
        img_x_max = slice_coords[0,valid_dex].max()
        img_y_min = slice_coords[1,valid_dex].min()
        img_y_max = slice_coords[1,valid_dex].max()

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

        new_img = np.zeros((n_img_x, n_img_y), dtype=float)

        img_ix = (new_img_pts[0,:]/resolution-img_x_min).astype(int)
        img_iy = (new_img_pts[1,:]/resolution-img_y_min).astype(int)

        for i_pt in valid_dex[0]:
            ix = img_ix[i_pt]
            iy = img_iy[i_pt]

            a_x = new_allen_dexes[0,i_pt]
            a_y = new_allen_dexes[1,i_pt]
            a_z = new_allen_dexes[2,i_pt]
            if ix>=new_img.shape[0] or iy>=new_img.shape[1]:
                print(ix,iy,' -- ',a_z,a_y,a_x)

            new_img[ix,n_img_y-1-iy] = img_data[a_z, a_y, a_x]

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
