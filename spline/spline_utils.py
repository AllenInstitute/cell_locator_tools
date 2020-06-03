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
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @property
    def border_x(self):
        return self._border_x

    @property
    def border_y(self):
        return self._border_y

    @property
    def border_x_pixels(self):
        return self._border_x_pixels

    @property
    def border_y_pixels(self):
        return self._border_y_pixels

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

        self._border_x = border_x
        self._border_y = border_y

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

    def _get_cross(self, ix, iy, mask):

        if iy not in self._by_y_lookup or ix not in self._by_x_lookup:
            return np.array([]), np.array([])

        this_row = self._by_y_lookup[iy]
        this_row_x = self._border_x_pixels_by_y[this_row[0]:this_row[1]]

        valid_row = True
        if len(this_row_x) > 0:
            left_side = this_row_x[np.where(this_row_x<=ix)]
            if len(left_side)>0:
                left_side = left_side.max()
            else:
                valid_row = False

            right_side = this_row_x[np.where(this_row_x>=ix)]
            if len(right_side)>0:
                right_side = right_side.min()
            else:
                valid_row = False
        else:
            valid_row = False

        if valid_row:
            interesting_row = np.arange(left_side, right_side).astype(int)
            interesting_row = interesting_row[np.where(np.logical_not(mask[iy, left_side:right_side]))]
        else:
            interesting_row = np.array([]).astype(int)

        this_col = self._by_x_lookup[ix]
        this_col_y = self._border_y_pixels_by_x[this_col[0]:this_col[1]]

        valid_col = True
        if len(this_col_y)>0:
            top_side = this_col_y[np.where(this_col_y>=iy)]
            if len(top_side)>0:
                top_side = top_side.min()
            else:
                valid_col = False

            bottom_side = this_col_y[np.where(this_col_y<=iy)]
            if len(bottom_side)>0:
                bottom_side = bottom_side.max()
            else:
                valid_col = False
        else:
            valid_col = False

        if valid_col:
            interesting_col = np.arange(bottom_side, top_side, 1).astype(int)
            interesting_col = interesting_col[np.where(np.logical_not(mask[bottom_side:top_side, ix]))]
        else:
            interesting_col = np.array([]).astype(int)

        return interesting_row, interesting_col


    def _scan_mask(self, ix, iy, mask):
        self.n_scans += 1

        (interesting_row,
         interesting_col) = self._get_cross(ix, iy, mask)

        if len(interesting_row)>0:
            mask[iy, interesting_row] = True

        if len(interesting_col)>0:
            mask[interesting_col, ix] = True

        for ix_int in interesting_row:
            self._interesting_ix.append((ix_int, iy))
        for iy_int in interesting_col:
            self._interesting_iy.append((ix, iy_int))

        return None

    def _clean_list(self, raw_pts, axis):
        pts = np.zeros((len(raw_pts), 2), dtype=int)
        for ii, pp in enumerate(raw_pts):
            pts[ii,0] = pp[0]
            pts[ii,1] = pp[1]

        sorted_dex = np.argsort(pts[:,axis])
        pts = pts[sorted_dex,:]

        out_pts = []
        for v in np.unique(pts[:,axis]):
            valid = np.where(pts[:,axis]==v)[0]
            valid_pts = pts[valid,:]
            sorted_dex = np.argsort(valid_pts[:,1-axis])
            valid_pts = valid_pts[sorted_dex,:]
            d = np.diff(valid_pts[:,1-axis])
            to_keep = np.where(d>1)[0]
            for ii in to_keep:
                out_pts.append(valid_pts[ii,:])
            out_pts.append(valid_pts[-1,:])

        return out_pts


    def get_mask(self):

        t0 = time.time()
        mask = np.zeros((self._n_y_pixels, self._n_x_pixels), dtype=bool)

        self._interesting_ix = []
        self._interesting_iy = []

        #centroid_x = np.round(self._spline.x.sum()/(len(self._spline.x)*self.resolution)).astype(int)
        #centroid_y = np.round(self._spline.y.sum()/(len(self._spline.y)*self.resolution)).astype(int)

        cx = None
        cy = None
        for ix, iy in zip(self._border_x_pixels_by_x, self._border_x_pixels_by_y):
            for d in ((10,0), (-10,0), (0,10), (0, -10)):
                r, c = self._get_cross(ix+d[0], iy+d[1], mask)
                if len(r)>3 and len(c)>0:
                    cx = r[len(r)//2]
                    cy = iy
                elif len(c)>3 and len(r)>0:
                    cx = ix
                    cy = c[len(c)//2]

                if cx is not None:
                    break

            if cx is not None:
                break

        self._cx = cx
        self._cy = cy
        #print(r)
        #print(c)

        self.n_scans = 1
        self._scan_mask(cx, cy, mask)
        while True:
            if len(self._interesting_ix) == 0 and len(self._interesting_iy)==0:
                break
            if len(self._interesting_ix)>0:
                for ii in range(len(self._interesting_ix)-1,-1,-1):
                    p = self._interesting_ix.pop(ii)
                    self._scan_mask(p[0], p[1], mask)
                self._interesting_iy = self._clean_list(self._interesting_iy, 1)

            if len(self._interesting_iy)>0:
                for ii in range(len(self._interesting_iy)-1,-1,-1):
                    p = self._interesting_iy.pop(ii)
                    self._scan_mask(p[0], p[1], mask)
                self._interesting_ix = self._clean_list(self._interesting_ix, 0)

        mask[self._border_y_pixels, self._border_x_pixels] = True

        print('got mask in %e seconds -- %e (n_scans %d)' % (time.time()-t0, mask.sum(), self.n_scans))
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
