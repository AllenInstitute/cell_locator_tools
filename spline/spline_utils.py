import numpy as np
import time


class Spline2D(object):

    @property
    def x(self):
        return self._x_vals

    @property
    def y(self):
        return self._y_vals


    def __init__(self, x_vals, y_vals):
        self._x_vals = np.copy(x_vals)
        self._y_vals = np.copy(y_vals)
        self._x_coeffs = self._get_coefficients(x_vals)
        self._y_coeffs = self._get_coefficients(y_vals)

    def values(self, ii, t):
        t2 = t**2
        t3 = t**3
        v_x = self._x_coeffs[ii, 0] + self._x_coeffs[ii, 1]*t + self._x_coeffs[ii, 2]*t2 + self._x_coeffs[ii, 3]*t3
        v_y = self._y_coeffs[ii, 0] + self._y_coeffs[ii, 1]*t + self._y_coeffs[ii, 2]*t2 + self._y_coeffs[ii, 3]*t3
        return (v_x, v_y)

    def derivatives(self, ii, t):
        t2 = t**2
        vp_x = self._x_coeffs[ii, 1] + 2*self._x_coeffs[ii, 2]*t + 3*self._x_coeffs[ii, 3]*t2
        vp_y = self._y_coeffs[ii, 1] + 2*self._y_coeffs[ii, 2]*t + 3*self._y_coeffs[ii, 3]*t2
        return (vp_x, vp_y)

    def second_derivatives(self, ii, t):
        vpp_x = 2*self._x_coeffs[ii, 2] + 6*self._x_coeffs[ii, 3]*t
        vpp_y = 2*self._y_coeffs[ii, 2] + 6*self._y_coeffs[ii, 3]*t
        return (vpp_x, vpp_y)

    def _get_coefficients(self, ordered_vals):
        m, bc = self._construct_matrix(ordered_vals)
        n_pts = len(ordered_vals)
        coeffs = np.linalg.solve(m, bc)
        return coeffs.reshape((n_pts, 4))

    def _construct_matrix(self, ordered_vals):
        """
        ordered_vals is a numpy array of the specified values in their desired order
        """
        # coeff[i_pt*4+j] will be the t^j coefficient for the ith pt

        n_pts = len(ordered_vals)
        n_coeffs = 4*n_pts
        mat = np.zeros((n_coeffs, n_coeffs), dtype=float)
        boundary_conditions = np.zeros(n_coeffs, dtype=float)
        for i_pt in range(n_pts):
            # value of spline at t=0
            mat[i_pt,i_pt*4] = 1.0
            boundary_conditions[i_pt] = ordered_vals[i_pt]

            # value of spline at t=1
            if i_pt<n_pts-1:
                i2 = i_pt+1
            else:
                i2 = 0
            mat[n_pts+i_pt, i_pt*4:i_pt*4+4] = 1.0
            boundary_conditions[n_pts+i_pt] = ordered_vals[i2]

            # derivative constraints
            # derivative of i_pt at t=1
            # derviative of i2 at t=0

            irow = 2*n_pts+i_pt
            mat[irow, i2*4+1] = 1.0
            mat[irow, i_pt*4+1] = -1.0
            mat[irow, i_pt*4+2] = -2.0
            mat[irow, i_pt*4+3] = -3.0

            # second derivative constraints
            irow = 3*n_pts+i_pt
            mat[irow, i2*4+2] = 2.0
            mat[irow, i_pt*4+2] = -2.0
            mat[irow, i_pt*4+3] = -6.0

        return mat, boundary_conditions


class Annotation(object):

    @property
    def resolution(self):
        return self._resolution

    def __init__(self, x_vals, y_vals, resolution):
        t0 = time.time()
        self._spline = Spline2D(x_vals, y_vals)
        self._resolution = resolution
        border_x = []
        border_y = []
        n_segments = len(x_vals)
        for i1 in range(n_segments):
            if i1<n_segments-1:
                i2 = i1+1
            else:
                i2 = 0
            d_max = 10.0*resolution

            # sample each curve at a fine enough resolution
            # that we will get all of the border pixels
            t = np.arange(0.0, 1.01, 0.01)
            d_threshold = 0.25*self.resolution
            while d_max>d_threshold:
                xx, yy = self._spline.values(i1, t)
                dist = np.sqrt((xx[:-1]-xx[1:])**2 + (yy[:-1]-yy[1:])**2)
                d_max = dist.max()
                if d_max>d_threshold:
                   bad_dex = np.where(dist>d_threshold)
                   new_t = t[bad_dex]+0.5*(t[bad_dex[0]+1]-t[bad_dex])
                   t = np.sort(np.concatenate([t, new_t]))
            # shave the value at t=0 because it is equivalent
            # to next segment's t=0
            border_x.append(xx[:-1])
            border_y.append(yy[:-1])
        border_x = np.concatenate(border_x)
        border_y = np.concatenate(border_y)

        # convert to integer pixel values
        self._border_x_pixels = np.round(border_x/resolution).astype(int)
        self._border_y_pixels = np.round(border_y/resolution).astype(int)
        #print(len(self._border_x_pixels))

        x_min = self._border_x_pixels.min()
        y_min = self._border_y_pixels.min()
        x_max = self._border_x_pixels.max()
        y_max = self._border_y_pixels.max()

        self._border_x_pixels -= x_min
        self._border_y_pixels -= y_min
        self._n_x_pixels = x_max-x_min+1
        self._n_y_pixels = y_max-y_min+1

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max

        # cull pixels that are identical to their neighbor
        n_border = len(self._border_x_pixels)
        d_pixel = np.ones(n_border)  # so that we keep the [-1] pixel
        d_pixel[:-1] = np.sqrt((self._border_x_pixels[:-1]-self._border_x_pixels[1:])**2 +
                          (self._border_y_pixels[:-1]-self._border_y_pixels[1:])**2)

        #print(np.unique(d_pixel,return_counts=True))

        valid = np.where(d_pixel>1.0e-6)
        self._border_x_pixels = self._border_x_pixels[valid]
        self._border_y_pixels = self._border_y_pixels[valid]
        #print(len(self._border_x_pixels))
        #print(self._border_x_pixels.dtype)
        #print(self._border_y_pixels.dtype)

        sorted_dex = np.argsort(self._border_x_pixels)
        self._border_x_pixels_by_x = self._border_x_pixels[sorted_dex]
        self._border_y_pixels_by_x = self._border_y_pixels[sorted_dex]
        self._by_x_lookup = {}
        for ix in np.unique(self._border_x_pixels_by_x):
            valid = np.where(self._border_x_pixels_by_x==ix)
            self._by_x_lookup[ix] = (valid[0].min(), valid[0].max()+1)

        sorted_dex = np.argsort(self._border_y_pixels)
        self._border_x_pixels_by_y = self._border_x_pixels[sorted_dex]
        self._border_y_pixels_by_y = self._border_y_pixels[sorted_dex]
        self._by_y_lookup = {}
        for iy in np.unique(self._border_y_pixels_by_y):
            valid = np.where(self._border_y_pixels_by_y==iy)
            self._by_y_lookup[iy] = (valid[0].min(), valid[0].max()+1)

        print('%e seconds' % (time.time()-t0))

    def _scan_mask(self, ix, iy, mask):
        this_row = self._by_y_lookup[iy]
        this_row_x = self._border_x_pixels_by_y[this_row[0]:this_row[1]]

        left_side = this_row_x[np.where(this_row_x<=ix)].max()
        right_side = this_row_x[np.where(this_row_x>=ix)].min()

        interesting_row = np.arange(left_side, right_side).astype(int)
        interesting_row = interesting_row[np.where(np.logical_not(mask[iy, left_side:right_side]))]
        mask[iy, interesting_row] = True

        this_col = self._by_x_lookup[ix]
        this_col_y = self._border_y_pixels_by_x[this_col[0]:this_col[1]]
        top_side = this_col_y[np.where(this_col_y>=iy)].min()
        bottom_side = this_col_y[np.where(this_col_y<=iy)].max()

        interesting_col = np.arange(bottom_side, top_side, 1).astype(int)
        interesting_col = interesting_col[np.where(np.logical_not(mask[bottom_side:top_side, ix]))]
        mask[interesting_col, ix] = True


        #print(interesting_row,left_side,right_side)
        #print(interesting_col,top_side, bottom_side)


        del this_row_x
        del this_col_y
        del this_row
        del this_col
        del left_side
        del right_side
        del top_side
        del bottom_side

        for ix_int in interesting_row:
            self._scan_mask(ix_int, iy, mask)

        del interesting_row

        for iy_int in interesting_col:
            self._scan_mask(ix, iy_int, mask)

        del interesting_col

        return None


    def get_mask(self):
        t0 = time.time()
        mask = np.zeros((self._n_y_pixels, self._n_x_pixels), dtype=bool)

        centroid_x = np.round(self._spline.x.sum()/(len(self._spline.x)*self.resolution)).astype(int)
        centroid_y = np.round(self._spline.y.sum()/(len(self._spline.y)*self.resolution)).astype(int)
        self._scan_mask(centroid_x-self._x_min, centroid_y-self._y_min, mask)
        print('got mask in %e seconds' % (time.time()-t0))
        return mask


if __name__ == "__main__":

    # test that matrix and bc get created correctly
    xx = np.array([1,2,3])
    yy = np.array([-4.0, -5.0, -1.0])
    s = Spline2D(xx, yy)
    m, b = s._construct_matrix(xx)
    np.testing.assert_array_equal(b,
                                  np.array([1,2,3,2,3,1,0,0,0,0,0,0]))

    m_control = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0,0,0],
                          [1,1,1,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,1,1,1],
                          [0,-1,-2,-3,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,-1,-2,-3,0,1,0,0],
                          [0,1,0,0,0,0,0,0,0,-1,-2,-3],
                          [0,0,-2,-6,0,0,2,0,0,0,0,0],
                          [0,0,0,0,0,0,-2,-6,0,0,2,0],
                          [0,0,2,0,0,0,0,0,0,0,-2,-6]])

    np.testing.assert_array_equal(m, m_control)


    r = s.values(0, 0)
    np.testing.assert_allclose(np.array(r), np.array((xx[0], yy[0])),
                               atol=1.0e-10, rtol=1.0e-10)
    r = s.values(0, 1)
    np.testing.assert_allclose(np.array(r), np.array((xx[1], yy[1])),
                               atol=1.0e-10, rtol=1.0e-10)
    r = s.values(1, 0)
    np.testing.assert_allclose(np.array(r), np.array((xx[1], yy[1])),
                               atol=1.0e-10, rtol=1.0e-10)

    r = s.values(1, 1)
    np.testing.assert_allclose(np.array(r), np.array((xx[2], yy[2])),
                               atol=1.0e-10, rtol=1.0e-10)
    r = s.values(2, 0)
    np.testing.assert_allclose(np.array(r), np.array((xx[2], yy[2])),
                               atol=1.0e-10, rtol=1.0e-10)

    r = s.values(2, 1)
    np.testing.assert_allclose(np.array(r), np.array((xx[0], yy[0])),
                               atol=1.0e-10, rtol=1.0e-10)

    for i1, i2 in zip((0, 1, 2), (1, 2, 0)):
        p1 = s.derivatives(i1, 1)
        p2 = s.derivatives(i2, 0)
        np.testing.assert_allclose(np.array(p1), np.array(p2), atol=1.0e-10, rtol=1.0e-10)
        pp1 = s.second_derivatives(i1, 1)
        pp2 = s.second_derivatives(i2, 0)
        np.testing.assert_allclose(np.array(pp1), np.array(pp2), atol=1.0e-10, rtol=1.0e-10)
