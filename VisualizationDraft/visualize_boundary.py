import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK
import cell_locator_utils

import os
import sys
this_dir = os.path.dirname(os.path.abspath('__file__'))
mod_dir = this_dir.replace('VisualizationDraft', 'spline')
sys.path.append(mod_dir)

import numpy as np
import json
import spline_utils
import time

import argparse

def get_boundary(brain_slice, markup_pts, threshold_factor):
    t0 = time.time()
    markup_slice_coords = brain_slice.coord_converter.c_to_slice(markup_pts)
    markup_slice_pixels = brain_slice.slice_to_pixel(markup_slice_coords[:2,:])

    annotation = spline_utils.Annotation(markup_slice_coords[0,:]-brain_slice.x_min,
                                         markup_slice_coords[1,:]-brain_slice.y_min,
                                         brain_slice.resolution*args.ds)

    annotation_mask = annotation.get_mask(just_boundary=True, threshold_factor=threshold_factor)
    print('full mask in %e seconds' % (time.time()-t0))
    return annotation_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None)
    parser.add_argument('--pts', action='store_true', default=False)
    parser.add_argument('--outname', type=str, default='annotation.pdf')
    parser.add_argument('--ds', type=float, default=0.2)
    args = parser.parse_args()

    if args.annotation is None:
        raise RuntimeError("must specify annotation")

    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)

    brain_img = cell_locator_utils.BrainImage(img_data, resolution)
    (slice_img,
     brain_slice_matrix) = brain_img.slice_img_from_annotation(args.annotation,
                                                               from_pts=False)

    (slice_img,
     brain_slice_pts) = brain_img.slice_img_from_annotation(args.annotation,
                                                            from_pts=True)


    with open(args.annotation, 'rb') as in_file:
        full_annotation = json.load(in_file)
    markup = full_annotation['Markups'][0]
    markup_pts = np.zeros((3,len(markup['Points'])), dtype=float)
    for i_p, p in enumerate(markup['Points']):
        markup_pts[0,i_p] = p['x']
        markup_pts[1,i_p] = p['y']
        markup_pts[2,i_p] = p['z']

    bdry_matrix_25 = get_boundary(brain_slice_matrix, markup_pts, threshold_factor=0.9)
    bdry_matrix = get_boundary(brain_slice_matrix, markup_pts, threshold_factor=0.05)
    print('')
    bdry_matrix = bdry_matrix.astype(int)*2
    bdry_matrix[bdry_matrix_25] -=1


    bdry_pts_25 = get_boundary(brain_slice_pts, markup_pts, threshold_factor=0.9)
    bdry_pts = get_boundary(brain_slice_pts, markup_pts, threshold_factor=0.05)
    bdry_pts = bdry_pts.astype(int)*2
    bdry_pts[bdry_pts_25] -=1

    bdry_pts = bdry_pts.transpose()

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(bdry_matrix)
    plt.subplot(1,2,2)
    plt.imshow(bdry_pts)
    plt.savefig(args.outname)
    print(bdry_pts.shape,bdry_pts.sum())
    print(bdry_matrix.shape,bdry_matrix.sum())
