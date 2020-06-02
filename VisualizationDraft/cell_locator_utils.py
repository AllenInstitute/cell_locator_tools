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

        self.allen_coords = np.zeros((3,self.nx0*self.ny0*self.nz0), dtype=float)

        mesh = np.meshgrid(resolution*np.arange(self.nx0),
                           resolution*np.arange(self.ny0),
                           resolution*np.arange(self.nz0),
                           indexing = 'ij')

        self.allen_coords[2,:] = mesh.pop(2).flatten()
        self.allen_coords[1,:] = mesh.pop(1).flatten()
        self.allen_coords[0,:] = mesh.pop(0).flatten()
        self.resolution = resolution

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

        valid_dex = np.where(coord_converter.get_slice_mask_from_allen(self.allen_coords,
                                                                       self.resolution))

        # find the coordinates of all of the voxels in the slice frame
        slice_coords = coord_converter.allen_to_slice(self.allen_coords[:,valid_dex[0]])

        # construct an empty grid to represent the 2D image of the slice
        img_x_min = slice_coords[0,:].min()
        img_x_max = slice_coords[0,:].max()
        img_y_min = slice_coords[1,:].min()
        img_y_max = slice_coords[1,:].max()

        img_x_min = np.round(img_x_min/self.resolution).astype(int)
        img_x_max = np.round(img_x_max/self.resolution).astype(int)
        img_y_min = np.round(img_y_min/self.resolution).astype(int)
        img_y_max = np.round(img_y_max/self.resolution).astype(int)

        n_img_cols = img_x_max-img_x_min+1
        n_img_rows = img_y_max-img_y_min+1

        new_img_pts = np.zeros((2, n_img_cols*n_img_rows), dtype=float)

        # fill new_img_pts with the 2D coordinates of the slice image
        pixel_mesh = np.meshgrid(np.arange(n_img_cols).astype(int),
                                 np.arange(n_img_rows).astype(int),
                                 indexing='ij')

        # pixel coordinates in new grid
        img_iy = pixel_mesh.pop(1).flatten()
        img_ix = pixel_mesh.pop(0).flatten()

        # world coordinates in slice frame
        new_img_pts[1,:] = self.resolution*(img_iy+img_y_min)
        new_img_pts[0,:] = self.resolution*(img_ix+img_x_min)

        # find the 3D voxels that actually fall within the 2D slice
        new_allen_coords = coord_converter.slice_to_allen(new_img_pts)
        new_allen_dexes = np.round(new_allen_coords/self.resolution).astype(int)

        valid_dex = np.where(np.logical_and(new_allen_dexes[0,:]>=0,
                             np.logical_and(new_allen_dexes[0,:]<self.nx0,
                             np.logical_and(new_allen_dexes[1,:]>=0,
                             np.logical_and(new_allen_dexes[1,:]<self.ny0,
                             np.logical_and(new_allen_dexes[2,:]>=0,
                                            new_allen_dexes[2,:]<self.nz0))))))

        ix_arr = img_ix[valid_dex]
        iy_arr = n_img_rows-1-img_iy[valid_dex]
        new_img_dex_flat = iy_arr*n_img_cols+ix_arr

        # get the pixel indices of the 3D voxels that are in the slice
        ax_arr = new_allen_dexes[0,valid_dex]
        ay_arr = new_allen_dexes[1,valid_dex]
        az_arr = new_allen_dexes[2,valid_dex]

        # get the image values from the atlas data and create a new image
        img_dex_flat = az_arr*(self.nx0*self.ny0)+ay_arr*self.nx0+ax_arr

        return img_dex_flat[0], new_img_dex_flat, n_img_cols, n_img_rows


    def slice_img_from_annotation(self, annotation_fname, from_pts=False):

        with open(annotation_fname, 'rb') as in_file:
            annotation = json.load(in_file)
        if from_pts:
            annotation = annotation['Markups'][0]


        coord_converter = CellLocatorTransformation(annotation, from_pts=from_pts)
        (img_dex_flat,
         new_img_dex_flat,
                  n_img_cols,
                  n_img_rows) = self.pixel_mask_from_CellLocatorTransformation(coord_converter)

        pixel_vals = self.img_data[img_dex_flat]
        new_img = np.zeros(n_img_rows*n_img_cols, dtype=float)
        new_img[new_img_dex_flat] = pixel_vals
        new_img = new_img.reshape(n_img_rows, n_img_cols)

        return new_img
