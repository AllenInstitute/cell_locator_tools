import os
import sys

import json
import numpy as np

import coords
import planar_geometry
import cell_locator_utils
import spline_utils


class VoxelMask(object):
    """
    This class stores a 3D grid of voxels as a flattened numpy array
    (i.e. the result of np.zeros((nx, ny, nz)).flatten())

    The method get_voxel_mask takes a CellLocator annotation and returns
    a boolean mask on that flattened voxel grid in which every voxel
    overlapping the annotation is set to True.
    """

    def __init__(self, nx, ny, nz, resolution):
        """
        Parameters
        ----------
        nx - an int; the number of voxels in the grid in the x direction
        ny - an int; the number of voxels in the grid in the y direction
        nz - an int; the number of voxels in the grid in a the z direction
        resolution - a float; the size in microns of a voxel's edge

        (all voxels are assumed to be cubes)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.resolution =resolution
        bdry = self._get_bdry()
        self.bdry_vol_coords = self._dex_to_vol(bdry)

    def _get_annotation(self, wc_origin, ann_pts, ann_class):
        slice_to_pixel = coords.PixelTransformer(wc_origin,
                                                 np.identity(2, dtype=float),
                                                 self.resolution,
                                                 self.resolution)

        annotation = ann_class(ann_pts[0,:], ann_pts[1,:],
                               pixel_transformer=slice_to_pixel)
        return annotation

    def _get_bdry(self):
        """
        Return the indices of all of the voxels on the boundary of the grid
        """
        z_range = np.arange(self.nz, dtype=int)
        y_range = np.arange(self.ny, dtype=int)
        x_range = np.arange(self.nx, dtype=int)
        bdry = []
        mesh = np.meshgrid(y_range, x_range, indexing='ij')
        y = mesh[0].flatten()
        x = mesh[1].flatten()
        del mesh
        xy = y*self.nx+x
        del x
        del y
        bdry.append(xy)
        bdry.append((self.nz-1)*self.nx*self.ny+xy)
        #assert np.max(bdry[-1])<nx*ny*nz
        del xy
        mesh = np.meshgrid(y_range, z_range, indexing='ij')
        y = mesh[0].flatten()
        z = mesh[1].flatten()
        del mesh
        zy = z*self.nx*self.ny+y*self.nx
        del z
        del y
        bdry.append(zy)
        bdry.append(zy+self.nx-1)
        #assert np.max(bdry[-1])<nx*ny*nz
        del zy
        mesh = np.meshgrid(x_range, z_range, indexing='ij')
        x = mesh[0].flatten()
        z = mesh[1].flatten()
        del mesh
        xz = z*self.nx*self.ny+x
        del x
        del z
        bdry.append(xz)
        bdry.append(xz+(self.ny-1)*self.nx)
        #assert np.max(bdry[-1])<nx*ny*nz
        del xz
        bdry = np.concatenate(bdry)
        #assert bdry.max()<nx*ny*nz
        return bdry

    def _dex_to_vol(self, dex):
        z = self.resolution*(dex//(self.nx*self.ny))
        y = self.resolution*((dex%(self.ny*self.nx))//self.nx)
        x = self.resolution*(dex%self.nx)
        return np.array([x, y, z])

    def get_voxel_mask(self, markup):
        """
        Read in a CellLocator annotation and return a boolean mask
        indicating which voxels in the flattened 3D grid are contained
        in the annotation.

        Parameters
        ----------
        markup - a dict denoting one CellLocator annotation. Read in from the
                 CellLocator json output

        Returns
        -------
        valid_voxels - a numpy array of booleans that is nx*ny*nz long.
                       Every element in the array that is True corresponds
                       to a voxel that is in the annotation.

        Note: valid_voxels.reshape((nz, ny, nx)) will reconstitute
        the 3D numpy array in the same shape as the initial atlas file.
        """

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

        edge_coords = slice_transform.allen_to_slice(self.bdry_vol_coords)
        wc_origin = np.array([edge_coords[0,:].min(),
                              edge_coords[1,:].min()])

        markup_slice = slice_transform.c_to_slice(markup_pts)
        annotation = self._get_annotation(wc_origin,
                                          markup_slice,
                                          ann_class)
        raw_mask = annotation.get_mask(self.resolution)

        center_x = np.mean(markup_slice[0,:])
        center_y = np.mean(markup_slice[1,:])

        dsq = ((annotation.border_x-center_x)**2 +
               (annotation.border_y-center_y)**2)

        radius_sq = dsq.max()

        center_allen = slice_transform.slice_to_allen(np.array([[center_x],
                                                                [center_y]]))
        radius = self.resolution+np.sqrt(thickness**2+radius_sq)

        center_voxel = np.round(center_allen/self.resolution).astype(int)
        radius_voxel = np.ceil(radius/self.resolution).astype(int)

        xyz_min = center_voxel-radius_voxel
        xyz_max = center_voxel+radius_voxel

        xr = np.arange(max(0,xyz_min[0]), min(xyz_max[0],self.nx), dtype=int)
        yr = np.arange(max(0,xyz_min[1]), min(xyz_max[1],self.ny), dtype=int)
        zr = np.arange(max(0,xyz_min[2]), min(xyz_max[2],self.nz), dtype=int)
        mesh = np.meshgrid(zr, yr, xr)
        idx = mesh.pop(2).flatten()
        idy = mesh.pop(1).flatten()
        idz = mesh.pop(0).flatten()

        vol_coords_first_dex = np.array([idx*self.resolution,
                                         idy*self.resolution,
                                         idz*self.resolution])

        z_plane = np.dot(slice_transform._a_to_slice[2,:3],
                         vol_coords_first_dex) + slice_transform._a_to_slice[2,3]

        upper_lim = 0.5*np.sqrt(3.0)*self.resolution
        lower_lim = -1.0*thickness-upper_lim
        in_plane_mask = np.logical_and(z_plane<=upper_lim, z_plane>=lower_lim)

        idx = idx[in_plane_mask]
        idy = idy[in_plane_mask]
        idz = idz[in_plane_mask]
        vol_coords_in_plane = vol_coords_first_dex[:,in_plane_mask]

        del in_plane_mask
        del z_plane
        del vol_coords_first_dex

        slice_coords = slice_transform.allen_to_slice(vol_coords_in_plane)
        pixel_coords = annotation.wc_to_pixels(slice_coords[:2,:])

        max_x = pixel_coords[0,:].max()+1
        max_y = pixel_coords[1,:].max()+1
        pixel_mask = np.zeros((max_x, max_y), dtype=bool)
        raw_mask = raw_mask.transpose()
        pixel_mask[:raw_mask.shape[0], :raw_mask.shape[1]] = raw_mask
        pixel_mask = pixel_mask.flatten()
        test_pixel_indices = pixel_coords[0,:]*max_y
        test_pixel_indices += pixel_coords[1,:]
        valid_voxels = np.zeros(self.nx*self.ny*self.nz, dtype=bool)
        in_plane_dexes = idz*self.nx*self.ny+idy*self.nx+idx
        valid_voxels[in_plane_dexes] = pixel_mask[test_pixel_indices]

        return valid_voxels


if __name__ == "__main__":
    pass
