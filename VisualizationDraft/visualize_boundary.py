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

import argparse

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
     brain_slice) = brain_img.slice_img_from_annotation(args.annotation,
                                                        from_pts=args.pts)

    with open(args.annotation, 'rb') as in_file:
        full_annotation = json.load(in_file)
    markup = full_annotation['Markups'][0]
    markup_pts = np.zeros((3,len(markup['Points'])), dtype=float)
    for i_p, p in enumerate(markup['Points']):
        markup_pts[0,i_p] = p['x']
        markup_pts[1,i_p] = p['y']
        markup_pts[2,i_p] = p['z']

    markup_slice_coords = brain_slice.coord_converter.c_to_slice(markup_pts)
    print(np.abs(markup_slice_coords[2,:]).max())
    markup_slice_pixels = brain_slice.slice_to_pixel(markup_slice_coords[:2,:])

    print('x ',markup_slice_coords[0,:])
    print(brain_slice.x_min)
    print('y ',markup_slice_coords[1,:])
    print(brain_slice.y_min)
    print('')

    annotation = spline_utils.Annotation(markup_slice_coords[0,:]-brain_slice.x_min,
                                         markup_slice_coords[1,:]-brain_slice.y_min,
                                         brain_slice.resolution*args.ds)

    annotation_mask = annotation.get_mask(just_boundary=True)
    #annotation_mask = np.zeros(raw_annotation_mask.shape, dtype=bool)
    #annotation_mask = raw_annotation_mask
    #for ii in range(raw_annotation_mask.shape[0]):
    #    annotation_mask[raw_annotation_mask.shape[0]-1-ii,:] = raw_annotation_mask[ii,:]

    #if args.pts:
    #    annotation_mask = annotation_mask.transpose()

    plt.figure(figsize=(10,10))
    plt.imshow(annotation_mask)
    plt.savefig(args.outname)
