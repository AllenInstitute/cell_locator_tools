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
    args = parser.parse_args()

    if args.annotation is None:
        raise RuntimeError("must specify annotation")

    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)

    brain_img = cell_locator_utils.BrainVolume(img_data, resolution)
    slice_img = brain_img.slice_img_from_annotation(args.annotation,
                                                        from_pts=args.pts)

    with open(args.annotation, 'rb') as in_file:
        full_annotation = json.load(in_file)
    markup = full_annotation['Markups'][0]
    markup_pts = np.zeros((3,len(markup['Points'])), dtype=float)
    for i_p, p in enumerate(markup['Points']):
        markup_pts[0,i_p] = p['x']
        markup_pts[1,i_p] = p['y']
        markup_pts[2,i_p] = p['z']

    markup_slice_coords = slice_img.brain_slice.coord_converter.c_to_slice(markup_pts)
    markup_slice_pixels = slice_img.brain_slice.slice_to_pixel(markup_slice_coords[:2,:])

    annotation = spline_utils.Annotation(markup_slice_coords[0,:], markup_slice_coords[1,:])

    ann_mask = annotation.get_mask(slice_img.brain_slice.resolution,
                                   pixel_transformer=slice_img.brain_slice._slice_to_pixel_transformer)
    mesh = np.meshgrid(np.arange(ann_mask.shape[0]), np.arange(ann_mask.shape[1]))

    ann_mask_pix = np.array([mesh[1].flatten(), mesh[0].flatten()])
    ann_mask_wc = annotation.pixels_to_wc(ann_mask_pix)
    ann_mask_pix = slice_img.brain_slice.slice_to_pixel(ann_mask_wc)

    ann_sp_pix = slice_img.brain_slice.slice_to_pixel(np.array([annotation._spline.x,
                                                      annotation._spline.y]))

    np.testing.assert_array_equal(ann_sp_pix, markup_slice_pixels)

    t_mask = ann_mask

    val = slice_img.img.max()+2.0

    x0 = ann_mask_pix[0,:].min()
    x1 = ann_mask_pix[0,:].max()+1
    y0 = ann_mask_pix[1,:].min()
    y1 = ann_mask_pix[1,:].max()+1

    #for ix, iy in zip(ann_mask_pix[0,:], ann_mask_pix[1,:]):
    #    if t_mask[iy-y0,ix-x0]:
    #        slice_img.img[iy,ix] += val
    #        slice_img.img[iy,ix]*=0.5

    slice_img.apply_mask(annotation)

    for ix, iy in zip(ann_sp_pix[0,:], ann_sp_pix[1,:]):
        slice_img.img[iy,ix] = 3*val

    plt.figure(figsize=(15,15))
    plt.imshow(slice_img.img,zorder=1)
    plt.scatter(markup_slice_pixels[0,:], markup_slice_pixels[1,:],
                color='r', zorder=2, s=1, marker='o', alpha=0.5)
    plt.savefig(args.outname)
    plt.close()

    print(slice_img.brain_slice.origin)
    print(np.dot(slice_img.brain_slice.coord_converter._c_to_slice,
                 np.append(slice_img.brain_slice.origin,1.0)))

    plt.figure(figsize=(10,10))
    t_arr = np.arange(0.01,1.01,0.01)
    for ii in range(len(annotation._spline.x)):
        if ii%3==0:
            plt.subplot(3,3,(ii//3)+1)
            plt.title('%d' % ii)
        xx,yy = annotation._spline.values(ii,t_arr)
        plt.plot(xx,yy)
        plt.plot([xx[0],xx[-1]],[yy[0],yy[-1]], color='r', linestyle='--')
    plt.savefig('spline_segments.pdf')
    plt.close()
