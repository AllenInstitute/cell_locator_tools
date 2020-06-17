import os
import json
import numpy as np

import coords
import planar_geometry
import cell_locator_utils
import spline_utils


def lean_voxel_mask(markup, nx, ny, nz, resolution):

    output_mask = np.zeros(nx*ny*nz, dtype=bool)

    if markup['RepresentationType'] == 'spline':
        ann_class = spline_utils.SplineAnnotation
    elif markup['RepresentationType'] == 'polyline':
        ann_class = spline_utils.PolyLineAnnotation
    else:
        raise RuntimeError("RepresentationType: %s" %
                           markup['RepresentationType'])

    thickness = markup['Thickness']

    slice_transform = cell_locator_utils.CellLocatorTransformation(markup,
                                             from_pts=False,
                                             forced_origin=None)

    markup_pts = np.zeros((3,len(markup['Points'])), dtype=float)
    for i_pt, pt in enumerate(markup['Points']):
        markup_pts[0,i_pt] = pt['x']
        markup_pts[1,i_pt] = pt['y']
        markup_pts[2,i_pt] = pt['z']

    c_markup_pts = np.dot(slice_transform._c_to_a_transposition[:3,:3],
                          markup_pts)
    plane = planar_geometry.Plane.plane_from_many_points(c_markup_pts.transpose())

    slice_coords = slice_transform.c_to_slice(markup_pts)
    wc_origin = np.array([slice_coords[0,:].min()-10.0,
                          slice_coords[1,:].min()-10.0])
    slice_to_pixel = coords.PixelTransformer(wc_origin,
                                             np.identity(2, dtype=float),
                                             resolution, resolution)

    annotation = ann_class(slice_coords[0,:], slice_coords[1,:],
                           pixel_transformer=slice_to_pixel)

    mask2d = annotation.get_mask(resolution)

    pixel_mesh = np.meshgrid(np.arange(mask2d.shape[1], dtype=int),
                             np.arange(mask2d.shape[0], dtype=int))

    pixel_coords = np.array([pixel_mesh[0].flatten(),
                             pixel_mesh[1].flatten()])
    del pixel_mesh

    mask2d = mask2d.flatten()
    wc_coords = slice_to_pixel.pixels_to_wc(pixel_coords)
    allen_coords = slice_transform.slice_to_allen(wc_coords)
    new_coords = np.zeros(allen_coords.shape, dtype=float)
    dz = resolution
    for zz in np.arange(resolution, -1.0*thickness, -resolution):
        for ii in range(3):
            new_coords[ii,:] = allen_coords[ii,:] + zz*plane.normal[ii]
        allen_pix = np.round(new_coords/resolution).astype(int)
        valid = np.logical_and(allen_pix[0,:]>=0,
                np.logical_and(allen_pix[0,:]<nx,
                np.logical_and(allen_pix[1,:]>=0,
                np.logical_and(allen_pix[1,:]<ny,
                np.logical_and(allen_pix[2,:]>=0,
                               allen_pix[2,:]<nz)))))
        allen_pix = allen_pix[:,valid]
        mask_values = mask2d[valid]
        allen_dex = allen_pix[2,:]*nx*ny+allen_pix[1,:]*nx+allen_pix[0,:]
        output_mask[allen_dex] = mask_values

    return output_mask #.reshape(nz, ny, nx)

if __name__ == "__main__":
    pass
