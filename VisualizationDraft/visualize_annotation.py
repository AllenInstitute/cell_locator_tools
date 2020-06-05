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

    #annotation = spline_utils.Annotation(markup_slice_coords[0,:]-brain_slice.x_min,
    #                                     markup_slice_coords[1,:]-brain_slice.y_min)


    annotation = spline_utils.Annotation(markup_slice_coords[0,:], markup_slice_coords[1,:])


    corner_pt = brain_slice.slice_to_pixel(np.array([[brain_slice.x_min],[brain_slice.y_min]]))

    ann_mask = annotation.get_mask(brain_slice.resolution)
    mesh = np.meshgrid(np.arange(ann_mask.shape[0]), np.arange(ann_mask.shape[1]))
    ann_mask_pix = np.array([mesh[0].flatten(), mesh[1].flatten()])
    ann_mask_wc = annotation.pixel_to_wc(ann_mask_pix)
    ann_mask_pix = brain_slice.slice_to_pixel(ann_mask_wc)

    t_mask = np.zeros(ann_mask.shape, dtype=bool)

    for ii in range(ann_mask.shape[0]):
        t_mask[ann_mask.shape[0]-1-ii,:] = ann_mask[ii,:]

    plt.figure(figsize=(15,15))
    #plt.subplot(1,2,1)

    #plt.imshow(slice_img)

    #plt.subplot(1,2,2)

    val = slice_img.max()+2.0

    print('val ',val)
    print('mask sum ',ann_mask.sum())
    print('img sum ',slice_img.sum())
    print(ann_mask_pix.shape)
    print(ann_mask_pix[0,:].min(),ann_mask_pix[0,:].max())
    print(ann_mask_pix[1,:].min(),ann_mask_pix[1,:].max())
    print(ann_mask.shape)
    print(slice_img.shape)

    x0 = ann_mask_pix[0,:].min()
    x1 = ann_mask_pix[0,:].max()+1
    y0 = ann_mask_pix[1,:].min()
    y1 = ann_mask_pix[1,:].max()+1

    slice_img[x0:x1, y0:y1][t_mask] += val
    slice_img[x0:x1, y0:y1][t_mask] *= 0.5

    print('img sum again ',slice_img.sum())
    #slice_img[ann_mask_pix[1,:], ann_mask_pix[0,:]][ann_mask] *= 0.5
    plt.imshow(slice_img,zorder=1)
    plt.scatter(markup_slice_pixels[0,:], markup_slice_pixels[1,:], color='r', zorder=2, s=0.25)

    plt.scatter(corner_pt[0,:], corner_pt[1,:], zorder=4, color='r', s=15)

    print((markup_slice_coords[0,:]-brain_slice.x_min).min()/brain_slice.resolution)
    print((markup_slice_coords[1,:]-brain_slice.y_min).min()/brain_slice.resolution)
    print(markup_slice_pixels[1,:].min())

    p=np.zeros((2,1),dtype=float)
    #p[0,0] = annotation.x_min*brain_slice.resolution+brain_slice.x_min
    #p[1,0] = annotation.y_min*brain_slice.resolution+brain_slice.y_min

    p[0,0] = markup_slice_coords[0,:].min()
    p[1,0] = markup_slice_coords[1,:].max()

    px = brain_slice.slice_to_pixel(p)
    print(px)
    print(annotation.y_max)

    plt.savefig(args.outname)
