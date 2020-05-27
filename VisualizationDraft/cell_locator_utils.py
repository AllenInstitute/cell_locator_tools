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
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:3,:] = pts
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._a_to_slice, pts_4d)
        return pts_4d[:3,:]

    def slice_to_allen(self, pts):
        """
        pts is a numpy array with shape (2, N) where N is the number of points
        """
        pts_4d = np.zeros((4,pts.shape[1]), dtype=float)
        pts_4d[:2,:] = pts
        pts_4d[2,:] = 0.0
        pts_4d[3,:] = 1.0
        pts_4d = np.dot(self._slice_to_a, pts_4d)
        return pts_4d[:3,:]
