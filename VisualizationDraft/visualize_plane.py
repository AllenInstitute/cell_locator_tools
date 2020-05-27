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


    valid_dex = np.where(np.abs(cell_locator_arr[2,:])<0.9*resolution)
    valid_pts = pt_arr[:,valid_dex]
    print('valid bounds')
    for ii in range(3):
        imin = valid_pts[ii,:].min()
        imax = valid_pts[ii,:].max()
        print('%d %d' % (np.round(imin/resolution).astype(int),
                         np.round(imax/resolution).astype(int)))

    print('')
    print('in new img space')
    for ii in range(2):
        imin = cell_locator_arr[ii,valid_dex].min()
        imax = cell_locator_arr[ii,valid_dex].max()
        print('%d %d' % (np.round(imin/resolution).astype(int),
                         np.round(imax/resolution).astype(int)))

    exit()

    pt = np.zeros(4,dtype=float)
    allen_pt = np.zeros(3, dtype=float)
    pt[3] = 1.0
    zmin = None
    xmin = None
    ymin = None
    for ix in range(img_data.shape[2]):
        for iy in range(img_data.shape[1]):
            for iz in range(img_data.shape[0]):
                allen_pt[0] = ix*resolution
                allen_pt[1] = iy*resolution
                allen_pt[2] = iz*resolution
                pt[:3] = allen_to_cell_locator(allen_pt)
                new_pt = np.dot(inverse_transform, pt)

                if zmin is None or np.abs(new_pt[2])<zmin:
                    zmin = np.abs(new_pt[2])
                if xmin is None or np.abs(new_pt[0])<xmin:
                    xmin = np.abs(new_pt[0])
                if ymin is None or np.abs(new_pt[1])<ymin:
                    ymin = np.abs(new_pt[1])
                if np.abs(new_pt[2])>0.9*resolution:
                    continue
                if px_min is None or new_pt[0]<px_min:
                    px_min = new_pt[0]
                if py_min is None or new_pt[1]<py_min:
                    py_min = new_pt[1]
                if px_max is None or new_pt[0]>px_max:
                    px_max = new_pt[0]
                if py_max is None or new_pt[1]>py_max:
                    py_max = new_pt[1]

    print('xmin ',xmin)
    print('ymin ',ymin)
    print('zmin ',zmin)

    print(px_min,px_max,py_min,py_max)
    print(px_min/resolution, px_max/resolution,
          py_min/resolution, py_max/resolution)

    nx_min = np.round(px_min/resolution).astype(int)
    nx_max = np.round(px_max/resolution).astype(int)
    ny_min = np.round(py_min/resolution).astype(int)
    ny_max = np.round(py_max/resolution).astype(int)
    found_ct = 0
    not_found_ct = 0
    new_img = np.zeros((nx_max-nx_min, ny_max-ny_min), dtype=float)

    newx_min = None
    newx_max = None
    newy_min = None
    newy_max = None
    newz_min = None
    newz_max = None

    pt = np.zeros(4, dtype=float)
    pt[3] = 1.0

    for ix in range(nx_max-nx_min):
        xx = nx_min+ix*resolution
        for iy in range(ny_max-ny_min):
            yy = ny_min+iy*resolution
            pt[0] = xx
            pt[1] = yy
            new_pt = np.dot(transform, pt)

            allen_pt = cell_locator_to_allen(new_pt)

            new_ix = np.round(allen_pt[0]/resolution).astype(int)
            new_iy = np.round(allen_pt[1]/resolution).astype(int)
            new_iz = np.round(allen_pt[2]/resolution).astype(int)

            if newx_min is None or new_ix<newx_min:
                newx_min = new_ix
            if newx_max is None or new_ix>newx_max:
                newx_max = new_ix
            if newy_min is None or new_iy<newy_min:
                newy_min = new_iy
            if newy_max is None or new_iy>newy_max:
                newy_max = new_iy
            if newz_min is None or new_iz<newz_min:
                newz_min = new_iz
            if newz_max is None or new_iz>newz_max:
                newz_max = new_iz


            is_found = False
            if new_ix>=0 and new_ix<img_data.shape[2]:
                if new_iy>=0 and new_iy<img_data.shape[1]:
                    if new_iz>=0 and new_iz<img_data.shape[0]:
                        is_found = True
                        #print(ix,iy,new_iz,new_iy,new_ix)
                        new_img[ix, iy] = img_data[new_iz,new_iy,new_ix]

            if is_found:
                found_ct += 1
            else:
                not_found_ct += 1

    print('found %d pixels (failed %d)' % (found_ct, not_found_ct))
    print('pixel val range ',new_img.max(),new_img.min(),np.median(new_img))
    print('ix ',newx_min, newx_max)
    print('iy ',newy_min, newy_max)
    print('iz ',newz_min, newz_max)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.show()
