import os
import numpy as np
import json

class CellLocatorTransformation(object):

    def __init__(self, annotation):
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

        # matrices to go between x,y,z (in CellLocator coordinates)
        # to x, y, z with z along the normal of the plane
        self._slice_to_c = np.zeros((4,4), dtype=float)
        if 'SplineOrientation' in annotation:
            orientation = annotation['SplineOrientation']
        else:
            orientation = annotation['DefaultSplineOrientation']

        for irow in range(4):
            for icol in range(4):
                self._slice_to_c[irow, icol] = orientation[irow*4+icol]

        self._c_to_slice = np.linalg.inv(self._slice_to_c)
        self._a_to_slice = np.dot(self._c_to_slice,
                                  self._a_to_c_transposition)
        self._slice_to_a = np.dot(self._c_to_a_transposition,
                                  self._slice_to_c)

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
        n_x -- number of rows in new_img.transpose()
        n_y -- number of cols in new_img.transpose()
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

        n_img_x = img_x_max-img_x_min+1
        n_img_y = img_y_max-img_y_min+1

        new_img_pts = np.zeros((2, n_img_x*n_img_y), dtype=float)

        # fill new_img_pts with the 2D coordinates of the slice image
        mesh = np.meshgrid(self.resolution*(img_x_min+np.arange(n_img_x)),
                           self.resolution*(img_y_min+np.arange(n_img_y)),
                           indexing='ij')

        new_img_pts[1,:] = mesh.pop(1).flatten()
        new_img_pts[0,:] = mesh.pop(0).flatten()

        # find the 3D voxels that actually fall within the 2D slice
        new_allen_coords = coord_converter.slice_to_allen(new_img_pts)
        new_allen_dexes = np.round(new_allen_coords/self.resolution).astype(int)

        valid_dex = np.where(np.logical_and(new_allen_dexes[0,:]>=0,
                             np.logical_and(new_allen_dexes[0,:]<self.nx0,
                             np.logical_and(new_allen_dexes[1,:]>=0,
                             np.logical_and(new_allen_dexes[1,:]<self.ny0,
                             np.logical_and(new_allen_dexes[2,:]>=0,
                                            new_allen_dexes[2,:]<self.nz0))))))

        # get the pixel indices of new_img_pts
        img_ix = (new_img_pts[0,:]/self.resolution-img_x_min).astype(int)
        img_iy = (new_img_pts[1,:]/self.resolution-img_y_min).astype(int)

        ix_arr = img_ix[valid_dex]
        iy_arr = n_img_y-1-img_iy[valid_dex]
        new_img_dex_flat = ix_arr*n_img_y+iy_arr

        # get the pixel indices of the 3D voxels that are in the slice
        ax_arr = new_allen_dexes[0,valid_dex]
        ay_arr = new_allen_dexes[1,valid_dex]
        az_arr = new_allen_dexes[2,valid_dex]

        # get the image values from the atlas data and create a new image
        img_dex_flat = az_arr*(self.nx0*self.ny0)+ay_arr*self.nx0+ax_arr

        return img_dex_flat, new_img_dex_flat, n_img_x, n_img_y


    def slice_img_from_annotation(self, annotation_fname):

        with open(annotation_fname, 'rb') as in_file:
            annotation = json.load(in_file)

        coord_converter = CellLocatorTransformation(annotation)
        (img_dex_flat,
         new_img_dex_flat,
                  n_img_x,
                  n_img_y) = self.pixel_mask_from_CellLocatorTransformation(coord_converter)

        pixel_vals = self.img_data[img_dex_flat]
        new_img = np.zeros(n_img_x*n_img_y, dtype=float)
        new_img[new_img_dex_flat] = pixel_vals
        new_img = new_img.reshape(n_img_x, n_img_y)

        new_img = new_img.transpose()
        return new_img
