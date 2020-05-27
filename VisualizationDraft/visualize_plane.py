import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import SimpleITK

import json
import time


def allen_to_cell_locator(pt):
    return np.array([pt[2], -pt[0], -pt[1]])

allen_to_cell_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])

def cell_locator_to_allen(pt):
    return np.array([-pt[1], -pt[2], pt[0]])

cell_to_allen_mat = np.array([[0.0, -1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])

if __name__ == "__main__":

    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)
    print(img_data.shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None)
    args = parser.parse_args()
    if not os.path.isfile(args.annotation):
        raise RuntimeError("%s is not a file" % args.annotation)

    with open(args.annotation, 'rb') as in_file:
        annotation = json.load(in_file)
    orientation = annotation['DefaultSplineOrientation']
    transform = np.zeros((4,4), dtype=float)
    for irow in range(4):
        for icol in range(4):
            transform[irow, icol] = orientation[irow*4+icol]

    inverse_transform = np.linalg.inv(transform)

    # testing block
    in_pt = np.zeros(4, dtype=float)
    in_pt[3] = 1.0
    c_pt = np.zeros(4, dtype=float)
    c_pt[3] = 1.0
    rng = np.random.RandomState(771)
    for ii in range(100):
        in_pt[:3] = 10000.0*(rng.random_sample(3)-0.5)
        c_pt[:3] = allen_to_cell_locator(in_pt)
        n_pt = np.dot(allen_to_cell_mat, in_pt)
        np.testing.assert_allclose(n_pt, c_pt, rtol=1.0e-10, atol=1.0e-10)
        t_pt = np.dot(inverse_transform, c_pt)
        assert np.abs(t_pt[3]-1.0)<1.0e-10
        b_pt = np.dot(transform, t_pt)
        np.testing.assert_allclose(c_pt, b_pt, rtol=1.0e-10, atol=1.0e-10)
        out_pt = cell_locator_to_allen(b_pt[:3])
        n_pt = np.dot(cell_to_allen_mat, b_pt)
        np.testing.assert_allclose(n_pt, in_pt, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(out_pt, in_pt[:3], rtol=1.0e-10, atol=1.0e-10)

    px_max = None
    py_max = None
    px_min = None
    py_min = None

    t0 = time.time()
    nx0 = img_data.shape[2]
    ny0 = img_data.shape[1]
    nz0 = img_data.shape[0]
    already_set = set()
    pt_arr = np.zeros((4,nx0*ny0*nz0), dtype=float)
    for ix in range(nx0):
        for iy in range(ny0):
            for iz in range(nz0):
                pt_dex = ix*(ny0*nz0)+iy*(nz0)+iz
                assert pt_dex not in already_set
                already_set.add(pt_dex)
                pt_arr[0,pt_dex] = ix*resolution
                pt_arr[1,pt_dex] = iy*resolution
                pt_arr[2,pt_dex] = iz*resolution
                pt_arr[3,pt_dex] = 1.0
    print('got raw grid in %e sec' % (time.time()-t0))

    # convert to cell locator
    cell_locator_arr = np.dot(allen_to_cell_mat, pt_arr)
    print('converted to cell locator after %e sec' % (time.time()-t0))

    # appy invers transform
    cell_locator_arr = np.dot(inverse_transform, cell_locator_arr)
    print('applied inverse_transform after %e sec' % (time.time()-t0))


    t_arr = np.dot(cell_to_allen_mat,np.dot(transform, cell_locator_arr))
    np.testing.assert_allclose(t_arr, pt_arr, rtol=1.0e-10, atol=1.0e-10)

    valid_dex = np.where(np.abs(cell_locator_arr[2,:])<0.5*resolution)
    valid_pts = pt_arr[:,valid_dex]
    print('valid bounds')
    for ii in range(3):
        imin = valid_pts[ii,:].min()
        imax = valid_pts[ii,:].max()
        print('%d %d' % (np.round(imin/resolution).astype(int),
                         np.round(imax/resolution).astype(int)))

    #print('assess grid-like nature')
    #x = np.round(valid_pts[1,:]/resolution).astype(int)
    #y = np.round(valid_pts[2,:]/resolution).astype(int)
    #for i1 in range(319):
    #    chosen = np.where(x==i1)
    #    y_chosen = np.sort(y[chosen])
    #    d = np.diff(y_chosen)
    #    print(i1,np.unique(d),y_chosen.min(),y_chosen.max())

    print('')
    print('in new img space')
    for ii in range(2):
        imin = cell_locator_arr[ii,valid_dex].min()
        imax = cell_locator_arr[ii,valid_dex].max()
        print('%d %d' % (np.round(imin/resolution).astype(int),
                         np.round(imax/resolution).astype(int)))

    img_x_min = cell_locator_arr[0,valid_dex].min()
    img_x_max = cell_locator_arr[0,valid_dex].max()
    img_y_min = cell_locator_arr[1,valid_dex].min()
    img_y_max = cell_locator_arr[1,valid_dex].max()

    img_x_min = np.round(img_x_min/resolution).astype(int)
    img_x_max = np.round(img_x_max/resolution).astype(int)
    img_y_min = np.round(img_y_min/resolution).astype(int)
    img_y_max = np.round(img_y_max/resolution).astype(int)

    n_img_x = img_x_max-img_x_min+1
    n_img_y = img_y_max-img_y_min+1

    new_img = np.zeros((n_img_x, n_img_y), dtype=float)

    new_img_pts = np.zeros((4, n_img_x*n_img_y), dtype=float)
    for ix in range(n_img_x):
        for iy in range(n_img_y):
            pt_dex = ix*n_img_y+iy
            new_img_pts[3, pt_dex] = 1.0
            new_img_pts[2, pt_dex] = 0.0
            new_img_pts[1, pt_dex] = (img_y_min+iy)*resolution
            new_img_pts[0, pt_dex] = (img_x_min+ix)*resolution

    new_img_pts = np.dot(transform, new_img_pts)
    print('after new_img_pts multiplied by transform')
    for ii in range(4):
        imin = new_img_pts[ii,:].min()
        imax = new_img_pts[ii,:].max()
        print('%e %e' % (imin,imax))

    new_img_pts = np.dot(cell_to_allen_mat, new_img_pts)

    print('unique new_img_pts[3,:] ',np.unique(new_img_pts[3,:]))
    new_img_pts = np.round(new_img_pts/resolution).astype(int)
    print('new image points imin, imax')
    for ii in range(3):
        imin = new_img_pts[ii,:].min()
        imax = new_img_pts[ii,:].max()
        print('%d %d' % (imin, imax))

    valid_dex = np.where(np.logical_and(new_img_pts[0,:]>=0,
                         np.logical_and(new_img_pts[0,:]<nx0,
                         np.logical_and(new_img_pts[1,:]>=0,
                         np.logical_and(new_img_pts[1,:]<ny0,
                         np.logical_and(new_img_pts[2,:]>=0,
                                        new_img_pts[2,:]<nz0))))))

    print('n_valid %d' % len(valid_dex[0]))

    exit()
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.show()
