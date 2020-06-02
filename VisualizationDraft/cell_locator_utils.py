import os
import sys

this_dir = os.environ['PWD']
mod_dir = this_dir.replace('VisualizationDraft', 'geom_package')
sys.path.append(mod_dir)
mod_dir = this_dir.replace('VisualizationDraft', 'spline')
sys.path.append(mod_dir)

import planar_geometry
import spline_utils
import numpy as np
import json

class CellLocatorTransformation(object):

    def __init__(self, annotation, from_pts=False):
        """
        annotation is a dict containing the annotation
        """
        # matrices to handle the fact that, in CellLocator:
        # +x = right
        # +y = anterior
        # +z = superior
        #
        # 'a' means 'Allen' and 'c' means 'CellLocator' in the
        # names of these matrices
        self._a_to_c_transposition = np.array([[0.0, 0.0, 1.0, 0.0],
                                               [-1.0, 0.0, 0.0, 0.0],
                                               [0.0, -1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 1.0]])

        self._c_to_a_transposition = np.array([[0.0, -1.0, 0.0, 0.0],
                                               [0.0, 0.0, -1.0, 0.0],
                                               [1.0, 0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0, 1.0]])

        if from_pts:
            self._slice_to_c = self.slice_to_c_from_points(annotation)
        else:
            self._slice_to_c = self.slice_to_c_from_orientation(annotation)

        self._c_to_slice = np.linalg.inv(self._slice_to_c)
        self._a_to_slice = np.dot(self._c_to_slice,
                                  self._a_to_c_transposition)
        self._slice_to_a = np.dot(self._c_to_a_transposition,
                                  self._slice_to_c)


    def slice_to_c_from_orientation(self, annotation):
        """
        Get matrix for transforming from slice coordinate to CellLocator
        3D coordinates from annotation['SplineOrientation']
        """

        # matrices to go between x,y,z (in CellLocator coordinates)
        # to x, y, z with z along the normal of the plane
        slice_to_c = np.zeros((4,4), dtype=float)
        if 'SplineOrientation' in annotation:
            orientation = annotation['SplineOrientation']
        else:
            orientation = annotation['DefaultSplineOrientation']

        for irow in range(4):
            for icol in range(4):
                slice_to_c[irow, icol] = orientation[irow*4+icol]
        return slice_to_c

    def slice_to_c_from_points(self, markup):
        pts = []
        for obj in markup['Points']:
            pts.append(np.array([obj['x'], obj['y'], obj['z']]))
        slice_plane = planar_geometry.Plane.plane_from_many_points(pts)
        z_to_norm = planar_geometry.rotate_v_into_w_3d(np.array([0.0, 0.0, 1.0]),
                                                       slice_plane.normal)

        slice_to_c = np.zeros((4,4), dtype=float)
        slice_to_c[:3, :3] = z_to_norm
        slice_to_c[3,3] = 1.0
        rotation_dot_origin = np.dot(z_to_norm, slice_plane.origin)
        for ii in range(3):
            slice_to_c[ii, 3] += slice_plane.origin[ii]
        return slice_to_c

    def allen_to_slice(self, pts):
        """
        pts is a numpy array with shape (3, N) where N is the number of points
        pts contains the coordinates of the points
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:3,:] = pts
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._a_to_slice, pts_4d)
        return pts_4d[:3,:]

    def slice_to_allen(self, pts):
        """
        pts is a numpy array with shape (2, N) where N is the number of points
        pts contains the coordinates of the points
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:2,:] = pts
        pts_4d[2,:] = 0.0
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._slice_to_a, pts_4d)
        return pts_4d[:3,:]

    def get_slice_mask_from_allen(self, pts, resolution):
        """
        pts is a numpy array wth shape (3, N) where N is the number of points;
        it contains the x,y,z coordinate values of the allen pixels
        """
        z_value = np.dot(self._a_to_slice[2,:3], pts)+self._a_to_slice[2,3]
        return np.abs(z_value)<0.5*resolution


class BrainSlice(object):

    @property
    def coord_converter(self):
        return self._coord_converter

    @property
    def resolution(self):
        return self._resolution

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def x_min_pix(self):
        return self._x_min_pix

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @property
    def y_min_pix(self):
        return self._y_min_pix

    @property
    def n_rows(self):
        return self._n_rows

    @property
    def n_cols(self):
        return self._n_cols

    def __init__(self, coord_converter, resolution, brain_volume):
        """
        brain_volume is the 3xN numpy array of x,y,z coords of full brain voxels
        """
        self._coord_converter = coord_converter
        self._resolution = resolution

        # find all of the voxels that are actually in the slice
        valid_dex = np.where(self.coord_converter.get_slice_mask_from_allen(brain_volume,
                                                                            self.resolution))

        # find the coordinates of all of the voxels in the slice frame
        slice_coords = coord_converter.allen_to_slice(brain_volume[:,valid_dex[0]])

        # construct an empty grid to represent the 2D image of the slice
        self._x_min = slice_coords[0,:].min()
        self._x_max = slice_coords[0,:].max()
        self._y_min = slice_coords[1,:].min()
        self._y_max = slice_coords[1,:].max()

        x1 = np.round(self.x_max/self.resolution).astype(int)
        x0 = np.round(self.x_min/self.resolution).astype(int)
        self._x_min_pix = x0
        self._n_cols = x1-x0+1

        y1 = np.round(self.y_max/self.resolution).astype(int)
        y0 = np.round(self.y_min/self.resolution).astype(int)
        self._y_min_pix= y0
        self._n_rows = y1-y0+1

    def allen_to_pixel(self, allen_coords):
        """
        Convert a 3xN array of allen coordinates into pixel coordinates on the slice
        """
        valid_dex = np.where(self.coord_converter.get_slice_mask_from_allen(allen_coords,
                                                                            self.resolution))[0]

        pixel_coords = np.NaN*np.ones((2,allen_coords.shape[1]), dtype=int)
        slice_coords = self.coord_converter.allen_to_slice(allen_coords[:,valid_dex])

        xx = np.round(slice_coords[0,:]/self.resolution).astype(int)
        pixel_coords[0,valid_dex] = xx-self.x_min_pix

        yy = np.round(slice_coords[1,:]/self.resolution).astype(int)
        pixel_coords[1,valid_dex] = yy-self.y_min_pix

        return pixel_coords, valid_dex

    def pixel_to_slice(self, pixel_coords):
        slice_coords = np.zeros(pixel_coords.shape, dtype=float)
        slice_coords[0,:] = (pixel_coords[0,:]+self.x_min_pix)*self.resolution
        yy = self.n_rows-1-pixel_coords[1,:]+self.y_min_pix
        slice_coords[1,:] = yy*self.resolution
        return slice_coords



class BrainImage(object):

    def __init__(self, img_data, resolution):
        """
        img_data is a 3D numpy array with the pixel data from the Brain Atlas
        resolution is the pixel resolution value for img_data
        """

        self.nx0 = img_data.shape[2]
        self.ny0 = img_data.shape[1]
        self.nz0 = img_data.shape[0]
        self.img_data = img_data.flatten()

        self.brain_volume = np.zeros((3,self.nx0*self.ny0*self.nz0), dtype=float)

        mesh = np.meshgrid(resolution*np.arange(self.nx0),
                           resolution*np.arange(self.ny0),
                           resolution*np.arange(self.nz0),
                           indexing = 'ij')

        self.brain_volume[2,:] = mesh.pop(2).flatten()
        self.brain_volume[1,:] = mesh.pop(1).flatten()
        self.brain_volume[0,:] = mesh.pop(0).flatten()
        self.resolution = resolution

    def allen_to_voxel(self, allen_coords):
        pixel_coords = np.round(allen_coords/self.resolution).astype(int)

        valid_dex = np.where(np.logical_and(pixel_coords[0,:]>=0,
                             np.logical_and(pixel_coords[0,:]<self.nx0,
                             np.logical_and(pixel_coords[1,:]>=0,
                             np.logical_and(pixel_coords[1,:]<self.ny0,
                             np.logical_and(pixel_coords[2,:]>=0,
                                            pixel_coords[2,:]<self.nz0))))))[0]
        return pixel_coords, valid_dex


    def pixel_mask_from_CellLocatorTransformation(self, coord_converter):
        """
        Accept a CellLocatorTransformation

        Return
        ------
        img_dex_flat -- a np.array of the indices of voxels in the plane
        new_img_dex_flat -- a np.array of where they should go in the new image
        n_cols -- number of cols in new_img
        n_rows -- number of rows in new_img
        """

        brain_slice = BrainSlice(coord_converter, self.resolution, self.brain_volume)
        return self.pixel_mask_from_BrainSlice(brain_slice)


    def pixel_mask_from_BrainSlice(self, brain_slice):
        """
        Accept a BrainSlice

        Return
        ------
        img_dex_flat -- a np.array of the indices of voxels in the plane
        new_img_dex_flat -- a np.array of where they should go in the new image
        n_cols -- number of cols in new_img
        n_rows -- number of rows in new_img
        """
        # fill new_img_pts with the 2D coordinates of the slice image
        pixel_mesh = np.meshgrid(np.arange(brain_slice.n_cols).astype(int),
                                 np.arange(brain_slice.n_rows).astype(int),
                                 indexing='ij')

        # pixel coordinates in new grid
        img_iy = pixel_mesh.pop(1).flatten()
        img_ix = pixel_mesh.pop(0).flatten()

        # world coordinates in slice frame
        new_img_pts = brain_slice.pixel_to_slice(np.array([img_ix, img_iy]))

        # find the 3D voxels that actually fall within the 2D slice
        new_allen_coords = brain_slice.coord_converter.slice_to_allen(new_img_pts)
        new_allen_voxels, voxel_mask = self.allen_to_voxel(new_allen_coords)

        ix_arr = img_ix[voxel_mask]
        iy_arr = img_iy[voxel_mask]
        new_img_dex_flat = iy_arr*brain_slice.n_cols+ix_arr

        # get the pixel indices of the 3D voxels that are in the slice
        ax_arr = new_allen_voxels[0,voxel_mask]
        ay_arr = new_allen_voxels[1,voxel_mask]
        az_arr = new_allen_voxels[2,voxel_mask]

        # get the image values from the atlas data and create a new image
        img_dex_flat = az_arr*(self.nx0*self.ny0)+ay_arr*self.nx0+ax_arr

        return img_dex_flat, new_img_dex_flat, brain_slice.n_cols, brain_slice.n_rows


    def slice_img_from_annotation(self, annotation_fname, from_pts=False):

        with open(annotation_fname, 'rb') as in_file:
            annotation = json.load(in_file)
        if from_pts:
            annotation = annotation['Markups'][0]


        coord_converter = CellLocatorTransformation(annotation, from_pts=from_pts)
        brain_slice = BrainSlice(coord_converter, self.resolution, self.brain_volume)
        (img_dex_flat,
         new_img_dex_flat,
                  n_img_cols,
                  n_img_rows) = self.pixel_mask_from_BrainSlice(brain_slice)

        pixel_vals = self.img_data[img_dex_flat]
        new_img = np.zeros(n_img_rows*n_img_cols, dtype=float)
        new_img[new_img_dex_flat] = pixel_vals
        new_img = new_img.reshape(n_img_rows, n_img_cols)

        return new_img, brain_slice
