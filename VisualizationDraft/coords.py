import numpy as np

class PixelTransformer(object):
    """
    A class to govern the conversion between physical world coordinates
    and pixel coordinates
    """

    def __init__(self, origin, rotation, x_resolution, y_resolution):
        """
        Parameters
        ----------
        origin - a 2-element numpy array indicating the origin of the
        coordinate system

        rotation - a 2x2 numpy array encoding the rotation from
        world coordinates to pixel coordinates

        x_resolution - a float indicating the resolution of pixels in
        the x direction

        y_resolution - a float indicating the resolution of pixels in
        the y direction
        """
        self._origin = np.copy(origin)
        self._rotation = np.copy(rotation)
        self._inv_rotation = np.linalg.inv(rotation)
        self._x_resolution = x_resolution
        self._y_resolution = y_resolution
        self._origin_dot_rotation = np.dot(rotation, self._origin)

    def __eq__(self, other):
        eps = 1.0e-10
        if not np.allclose(self._origin, other._origin, rtol=eps, atol=eps):
            return False
        if not np.allclose(self._rotation, other._rotation, rtol=eps, atol=eps):
            return False
        if np.abs(self._x_resolution-other._x_resolution)>eps:
            return False
        if np.abs(self._y_resolution-other._y_resolution)>eps:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def wc_to_pixels(self, pts):
        """
        Convert an array of points in world coordinates into
        pixel coordinates

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
        Convert an array of points in pixel coordinates into
        world coordinates

        pixels is 2xN
        """
        w = np.copy(pixels).astype(float)
        w[0,:] *= self._x_resolution
        w[1,:] *= self._y_resolution
        w = np.dot(self._inv_rotation, w)
        w[0,:] += self._origin[0]
        w[1,:] += self._origin[1]
        return w
