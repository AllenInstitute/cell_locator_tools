import os
import sys

import planar_geometry
import spline_utils
import coords
import numpy as np
import json
import copy

import time

class CellLocatorTransformation(object):

    @property
    def origin(self):
        """
        in Cell Locator coordinates
        """
        return self._origin

    def __init__(self, annotation, from_pts=False, forced_origin=None):
        """
        annotation is a dict containing the annotation

        force_origin is in CellLocator coordinates
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

        if forced_origin is not None:
            self._slice_to_c[:3,3] = forced_origin

        # in 3-D cell-locator coordinates
        self._origin = np.array([self._slice_to_c[0,3],
                                 self._slice_to_c[1,3],
                                 self._slice_to_c[2,3]])

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
        """
        Get matrix for transforming from slice coordinate to CellLocator
        3D coordinates from annotation['Points']
        """
        pts = []
        for obj in markup['Points']:
            pts.append(np.array([obj['x'], obj['y'], obj['z']]))
        pts = np.array(pts)
        slice_plane = planar_geometry.Plane.plane_from_many_points(pts)
        z_to_norm = planar_geometry.rotate_v_into_w_3d(np.array([0.0, 0.0, 1.0]),
                                                       slice_plane.normal)

        # find 2-d principal axis of points in plane
        # rotate plane so that axis lines up with x axis
        oo = pts.sum(axis=0)/pts.shape[0]
        norm_to_z = np.linalg.inv(z_to_norm)
        dsq = np.sum((oo-pts)**2, axis=1)
        n = dsq.sum()
        pts_in_plane = np.dot(norm_to_z, pts.transpose())
        xbar = np.dot(dsq, pts_in_plane[0,:])
        ybar = np.dot(dsq, pts_in_plane[1,:])
        xybar = np.dot(dsq, pts_in_plane[0,:]*pts_in_plane[1,:])
        xsqbar = np.dot(dsq, pts_in_plane[0,:]**2)
        tan_theta = (xbar*ybar-n*xybar)/(xbar*xbar-n*xsqbar)
        cos_sq_theta = 1.0/(1.0+tan_theta**2)
        cos_theta = np.sqrt(cos_sq_theta)
        sin_theta = np.sign(tan_theta)*np.sqrt(1.0-cos_sq_theta)
        v = np.array([cos_theta, sin_theta, 0.0])
        rot_plane = planar_geometry.rotate_v_into_w_3d(v, np.array([1.0, 0.0, 0.0]))
        z_to_norm = np.dot(z_to_norm, rot_plane)

        slice_to_c = np.zeros((4,4), dtype=float)
        slice_to_c[:3, :3] = z_to_norm
        slice_to_c[3,3] = 1.0
        for ii in range(3):
            slice_to_c[ii, 3] += slice_plane.origin[ii]

        return slice_to_c

    def allen_to_c(self, pts):
        """
        Convert an array of points in Allen Institute coordinates
        into CellLocator coordinates.

        Parameters
        ----------
        pts is a numpy array with shape (3, N) where N is the number of points
        pts contains the coordinates of the points

        Returns
        -------
        a numpy array of shape (3, N)
        """
        return np.dot(self._a_to_c[:3,:3], pts)

    def c_to_allen(self, pts):
        """
        Convert an array of points in CellLocator coordinates
        to Allen Institute coordinates.

        Parameters
        ----------
        pts is a numpy array with shape (3, N) where N is the number of points
        pts contains the coordinates of the points

        Returns
        -------
        a numpy array with shape (3, N)
        """
        return np.dot(self._c_to_a[:3,:3], pts)

    def c_to_slice(self, pts):
        """
        Convert an array of points from CellLocator 3D
        coordinates to 3D coordinates with the plane of the
        slice at z=0

        Parameters
        ----------
        pts is a numpy array with shape (3, N) where N is the number of points
        pts contains the coordinates of the points

        Returns
        -------
        a numpy array of shape (3, N) denoting coordinates with the slice
        at z=0
        """
        pts_4d = np.zeros((4, pts.shape[1]), dtype=float)
        pts_4d[:3,:] = pts
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._c_to_slice, pts_4d)
        return pts_4d[:3,:]

    def allen_to_slice(self, pts):
        """
        Convert an array of points in Allen Institute coordinates into
        the 3D coordinate system in which the plane of the slice is at
        z = 0

        Parameters
        ----------
        pts is a numpy array with shape (3, N) where N is the number of points
        pts contains the coordinates of the points

        Returns
        -------
        A numpy array with shape (3, N)
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:3,:] = pts
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._a_to_slice, pts_4d)
        return pts_4d[:3,:]

    def slice_to_allen(self, pts):
        """
        Convert 2D coordinates in the plane of the slice into 3D coordinates
        in the Allen Institute system.

        Parameters
        ----------
        pts is a numpy array with shape (2, N) where N is the number of points
        pts contains the coordinates of the points

        Returns
        -------
        a numpy array with shape (3, N)
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:2,:] = pts
        pts_4d[2,:] = 0.0
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._slice_to_a, pts_4d)
        return pts_4d[:3,:]

    def z_from_allen(self, pts):
        """
        Calculate distance from the plane of the slice

        Parameters
        ----------
        pts is a numpy array wth shape (3, N) where N is the number of points;
        it contains the x,y,z coordinate values of the voxels in the Allen
        Institute system

        Returns
        -------
        a numpy array of z voxel values in the system where the slice is
        at z=0
        """
        return np.dot(self._a_to_slice[2,:3], pts)+self._a_to_slice[2,3]
