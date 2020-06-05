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

    annotation = spline_utils.Annotation(markup_slice_coords[0,:]-brain_slice.x_min,
                                         markup_slice_coords[1,:]-brain_slice.y_min)


    #annotation = spline_utils.Annotation(markup_slice_coords[0,:], markup_slice_coords[1,:])


    corner_pt = brain_slice.slice_to_pixel(np.array([[brain_slice.x_min],[brain_slice.y_min]]))

    #np.testing.assert_equal(annotation._spline.x, markup_slice_coords[0,:]-brain_slice.x_min)
    #np.testing.assert_equal(annotation._spline.y, markup_slice_coords[1,:]-brain_slice.y_min)

    raw_annotation_mask = annotation.get_mask(brain_slice.resolution)
    annotation_mask = np.zeros(raw_annotation_mask.shape, dtype=bool)
    #annotation_mask = raw_annotation_mask
    for ii in range(raw_annotation_mask.shape[0]):
        annotation_mask[raw_annotation_mask.shape[0]-1-ii,:] = raw_annotation_mask[ii,:]

    #print(annotation.x_min,slice_img.shape[1])
    #print(annotation.y_min,slice_img.shape[0])

    

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)

    plt.imshow(slice_img)

    plt.subplot(1,2,2)

    val = slice_img.max()+2.0
    dx = annotation_mask.shape[1]
    dy = annotation_mask.shape[0]
    minx = np.round(annotation.x_min/brain_slice.resolution).astype(int)
    maxx = minx+dx
    ymx = np.round(annotation.y_max/brain_slice.resolution).astype(int)
    miny = brain_slice.n_rows-1-ymx #annotation.y_min #brain_slice.n_rows-1+brain_slice.y_min_pix-annotation.y_min
    maxy=miny+dy

    print('val ',val)
    print('mask sum ',annotation_mask.sum())
    slice_img[miny:maxy,minx:maxx][annotation_mask] += val
    slice_img[miny:maxy,minx:maxx][annotation_mask] *= 0.5
    plt.imshow(slice_img,zorder=1)
    plt.scatter(markup_slice_pixels[0,:], markup_slice_pixels[1,:], color='r', zorder=2, s=0.25)

    plt.scatter([minx],[miny], color='y', zorder=3, s=10)

    plt.scatter(corner_pt[0,:], corner_pt[1,:], zorder=4, color='c', s=15)
 
    print('minxy ',minx,miny)
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
