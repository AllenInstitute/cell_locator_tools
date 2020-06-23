import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir.replace('VisualizationDraft','geom_package'))

import json
import numpy as np

import coords
import planar_geometry
import cell_locator_utils
import spline_utils

def _get_volume_coords(nx, ny, nz, resolution):

    mesh = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    vol_coords = np.zeros((3,nx*ny*nz), dtype=float)
    vol_coords[0,:] = mesh.pop(2).flatten()*resolution
    vol_coords[1,:] = mesh.pop(1).flatten()*resolution
    vol_coords[2,:] = mesh.pop(0).flatten()*resolution
    return vol_coords

def _get_annotation(wc_origin, resolution, ann_pts, ann_class):
    slice_to_pixel = coords.PixelTransformer(wc_origin,
                                             np.identity(2, dtype=float),
                                             resolution,
                                             resolution)

    annotation = ann_class(ann_pts[0,:], ann_pts[1,:],
                           pixel_transformer=slice_to_pixel)
    return annotation

def _get_bdry(nx, ny, nz):
    z_range = np.arange(nz, dtype=int)
    y_range = np.arange(ny, dtype=int)
    x_range = np.arange(nx, dtype=int)
    bdry = []
    mesh = np.meshgrid(y_range, x_range, indexing='ij')
    y = mesh[0].flatten()
    x = mesh[1].flatten()
    del mesh
    xy = y*nx+x
    del x
    del y
    bdry.append(xy)
    bdry.append((nz-1)*nx*ny+xy)
    assert np.max(bdry[-1])<nx*ny*nz
    del xy
    mesh = np.meshgrid(y_range, z_range, indexing='ij')
    y = mesh[0].flatten()
    z = mesh[1].flatten()
    del mesh
    zy = z*nx*ny+y*nx
    del z
    del y
    bdry.append(zy)
    bdry.append(zy+nx-1)
    assert np.max(bdry[-1])<nx*ny*nz
    del zy
    mesh = np.meshgrid(x_range, z_range, indexing='ij')
    x = mesh[0].flatten()
    z = mesh[1].flatten()
    del mesh
    xz = z*nx*ny+x
    del x
    del z
    bdry.append(xz)
    bdry.append(xz+(ny-1)*nx)
    assert np.max(bdry[-1])<nx*ny*nz
    del xz
    bdry = np.concatenate(bdry)
    assert bdry.max()<nx*ny*nz
    return bdry

def lean_voxel_mask(markup, nx, ny, nz, resolution, vol_coords=None):

    if vol_coords is None:
        vol_coords = _get_volume_coords(nx, ny, nz, resolution)

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

    z_plane = np.dot(slice_transform._a_to_slice[2,:3],
                     vol_coords) + slice_transform._a_to_slice[2,3]

    upper_lim = 0.5*np.sqrt(3.0)*resolution
    lower_lim = -1.0*thickness-upper_lim
    in_plane = np.logical_and(z_plane<=upper_lim, z_plane>=lower_lim)

    slice_coords = slice_transform.allen_to_slice(vol_coords[:,in_plane])

    wc_origin = np.array([slice_coords[0,:].min(),
                          slice_coords[1,:].min()])

    annotation = _get_annotation(wc_origin, resolution,
                                 slice_transform.c_to_slice(markup_pts),
                                 ann_class)

    pixel_coords = annotation.wc_to_pixels(slice_coords[:2,:])

    raw_mask = annotation.get_mask(resolution)

    max_x = pixel_coords[0,:].max()+1
    max_y = pixel_coords[1,:].max()+1
    pixel_mask = np.zeros((max_x, max_y), dtype=bool)
    raw_mask = raw_mask.transpose()
    pixel_mask[:raw_mask.shape[0], :raw_mask.shape[1]] = raw_mask
    pixel_mask = pixel_mask.flatten()
    test_pixel_indices = pixel_coords[0,:]*max_y
    test_pixel_indices += pixel_coords[1,:]
    valid_voxels = np.zeros(nx*ny*nz, dtype=bool)
    valid_voxels[in_plane] = pixel_mask[test_pixel_indices]

    return valid_voxels


if __name__ == "__main__":
    pass
