import local_utils
import sys
import unittest
import numpy as np
import coords


class PixelTransformTest(unittest.TestCase):

    def test_wc_to_pix(self):
        rng = np.random.RandomState(99)
        x_axis = rng.random_sample(2)-0.5
        x_axis /= np.sqrt(np.dot(x_axis, x_axis))
        y_axis = np.array([-x_axis[1], x_axis[0]])
        self.assertLess(np.abs(np.dot(x_axis, y_axis)), 1.0e-10)
        self.assertLess(np.abs(1.0-np.dot(y_axis, y_axis)), 1.0e-10)
        self.assertLess(np.abs(1.0-np.dot(x_axis, x_axis)), 1.0e-10)
        rot = np.array([x_axis,y_axis])
        x_resolution = 17.1
        y_resolution = 4.6
        origin = np.array([44.3, 22.1])
        transformer = coords.PixelTransformer(origin, rot,
                                              x_resolution, y_resolution)

        d_t = np.array([[1.2, 4.51, 6.1, -9.49, -11.6, 2.3],
                        [34.6, -3.2, -1.9, 2.1, 8.9, 4.2]])

        truth = np.array([[1,5,6,-9,-12,2],
                          [35, -3, -2, 2, 9, 4]])

        pts = np.zeros(d_t.shape, dtype=float)
        for ii in range(pts.shape[1]):
            p = origin + d_t[0,ii]*x_axis*x_resolution
            p += d_t[1,ii]*y_axis*y_resolution
            pts[:,ii] = p

        test = transformer.wc_to_pixels(pts)
        np.testing.assert_array_equal(test, truth)

        back_to_wc = transformer.pixels_to_wc(test)
        dd = back_to_wc-pts
        self.assertLess(np.abs(dd[0,:]/x_resolution).max(), 0.5)
        self.assertLess(np.abs(dd[1,:]/y_resolution).max(), 0.5)


if __name__ == "__main__":
    unittest.main()
