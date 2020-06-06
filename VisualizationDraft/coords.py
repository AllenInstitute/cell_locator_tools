import numpy as np

class PixelTransformer(object):

    def __init__(self, origin, rotation, x_resolution, y_resolution):
        self._origin = np.copy(origin)
        self._rotation = np.copy(rotation)
        self._inv_rotation = np.linalg.inv(rotation)
        self._x_resolution = x_resolution
        self._y_resolution = y_resolution
        self._origin_dot_rotation = np.dot(rotation, self._origin)

    def wc_to_pixels(self, pts):
        """
        pts is 2xN
        """
        r = np.dot(self._rotation, pts)
        r[0,:] -= self._origin_dot_rotation[0]
        r[1,:] -= self._origin_dot_rotation[1]
        r[0,:] /= self._x_resolution
        r[1,:] /= self._y_resolution
        return np.round(r).astype(int)

    def pixels_to_wc(self, pixels):
        """
        pixels is 2xN
        """
        w = np.copy(pixels).astype(float)
        w[0,:] *= self._x_resolution
        w[1,:] *= self._y_resolution
        w = np.dot(self._inv_rotation, w)
        w[0,:] += self._origin[0]
        w[1,:] += self._origin[1]
        return w
