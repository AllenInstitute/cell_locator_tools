import numpy as np
import coords
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


class AnnotationBase(object):

    @property
    def x_min(self):
        return self.coord_transform._origin[0]

    @property
    def y_min(self):
        return self.coord_transform._origin[1]

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
        return self.coord_transform._x_resolution

    def __init__(self, x_vals, y_vals, pixel_transformer=None):
        """
        Parameters
        ----------
        x_vals is a numpy array of the x values of the points defining
        the annotation

        y_vals is a numpy array of hte y values of the points defining
        the annotation

        pixel_transformer is an instance of coords.PixelTransformer that
        that converts from microns to pixel values in the slice
        """
        self._x_vals = np.copy(x_vals)
        self._y_vals = np.copy(y_vals)
        self._clean_border()
        self.coord_transform = pixel_transformer

    def wc_to_pixels(self, xy):
        """
        Convert from microns to pixels in the annotation

        xy is a 2xN numpy array
        """
        return self.coord_transform.wc_to_pixels(xy)

    def pixels_to_wc(self, pixels):
        """
        Convert from pixels to microns in the annotation

        pixels is a 2xN numpy array
        """
        return self.coord_transform.pixels_to_wc(pixels)

    def _clean_border(self):
        self._border_x = None
        self._border_y = None
        self._border_x_pixels = None
        self._border_y_pixels = None
        self._border_x_pixels_by_x = None
        self._border_y_pixels_by_x = None
        self._border_x_pixels_by_y = None
        self._border_y_pixels_by_y = None
        self._by_x_lookup = None
        self._by_y_lookup = None
        self._border_interpolator = None

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

        to_keep = self._border_mask[pts[:,1], pts[:,0]]

        pts = pts[to_keep,:]

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


    def _generate_mask(self, resolution,
                       just_boundary=False,
                       threshold_factor=0.25):

        t0 = time.time()
        self._build_boundary(resolution, threshold_factor)

        self._mask_params = {'resolution': resolution,
                             'just_boundary': just_boundary,
                             'threshold_factor': threshold_factor,
                             'pixel_transformer': self.coord_transform}

        mask = np.zeros((self._n_y_pixels, self._n_x_pixels), dtype=bool)

        # create a mask that is False on border pixels and True everywhere else;
        # this will be used to prevent border pixels from being selected as
        # "interesting" points when scanning
        self._border_mask = np.ones(mask.shape, dtype=bool)
        self._border_mask[self.border_y_pixels, self.border_x_pixels] = False

        self._interesting_ix = []
        self._interesting_iy = []

        cx = None
        cy = None
        for ix, iy in zip(self._border_x_pixels, self._border_y_pixels):
            for d in ((10,0), (-10,0), (0,10), (0, -10),
                      (10, 10), (10, -10), (-10, -10), (-10, 10)):
                r, c = self._get_cross(ix+d[0], iy+d[1], mask)
                if len(r)>3 and len(c)>0:
                    cx = r[len(r)//2]
                    cy = iy+d[1]
                elif len(c)>3 and len(r)>0:
                    cx = ix+d[0]
                    cy = c[len(c)//2]

                if cx is not None:
                    break

            if cx is not None:
                break

        self._cx = cx
        self._cy = cy

        self.n_scans = 1
        if not just_boundary:
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

        del self._border_mask

        return mask

    def get_mask(self, resolution, just_boundary=False, threshold_factor=0.25):
        """
        Return a 2D boolean mask of pixels that are inside the annotation.
        This mask is in the coordinate system of the brain slice denoted
        by the annotation.

        Parameters
        ----------
        resolution - the size in microns of one edge of a pixel
        (pixels are assumed to be square)

        just_boundary - a boolean; if True, only mask the pixels on the
        boundary of the annotation

        threshold_factor - a float; any pixel within threshold_factor*resolution
        of the boundary wil be masked

        Returns
        -------
        A 2D numpy array of booleans. True denotes pixels that are inside
        the annotation. False denotes pixels that are not.
        """
        must_generate = False
        if not hasattr(self, '_mask_params'):
            must_generate = True
        else:
            if self._mask_params['just_boundary'] != just_boundary:
                must_generate = True
            if np.abs(self._mask_params['resolution']-resolution)>1.0e-10:
                must_generate = True
            if np.abs(self._mask_params['threshold_factor']-threshold_factor)>1.0e-10:
                must_generate = True
            if self.coord_transform is not None and self._mask_params['pixel_transformer'] is None:
                must_generate = True
            elif self._mask_params['pixel_transformer'] is not None and self.coord_transform is None:
                must_generate = True
            elif self._mask_params['pixel_transformer'] != self.coord_transform:
                must_generate = True
        if must_generate:
            self._mask = self._generate_mask(resolution, just_boundary=just_boundary,
                                             threshold_factor=threshold_factor)
        return self._mask

    def _build_boundary(self, resolution, threshold_factor):
        """
        Set the member variables that store the lists of boundary pixels
        for the annotation
        """
        self._clean_border()

        (border_x,
         border_y) = self._build_raw_boundary(resolution, threshold_factor)

        just_made_coord_transform = False
        if self.coord_transform is None:
            just_made_coord_transform = True
            x0 = border_x.min()
            y0 = border_y.min()
            self.coord_transform = coords.PixelTransformer(np.array([x0, y0]),
                                                           np.identity(2,dtype=float),
                                                           resolution, resolution)

        self._border_x = border_x
        self._border_y = border_y

        # convert to integer pixel values
        pixel_coords = self.wc_to_pixels(np.array([border_x, border_y]))
        self._border_x_pixels = pixel_coords[0,:]
        self._border_y_pixels = pixel_coords[1,:]

        if just_made_coord_transform:
            self._n_x_pixels = self.border_x_pixels.max()-self.border_x_pixels.min()+1
            self._n_y_pixels = self.border_y_pixels.max()-self.border_y_pixels.min()+1
        else:
            self._n_x_pixels = self.border_x_pixels.max()+1
            self._n_y_pixels = self.border_y_pixels.max()+1


        # cull pixels that are identical to their neighbor
        n_border = len(self._border_x_pixels)
        d_pixel = np.ones(n_border)  # so that we keep the [-1] pixel
        d_pixel[:-1] = np.sqrt((self._border_x_pixels[:-1]-self._border_x_pixels[1:])**2 +
                          (self._border_y_pixels[:-1]-self._border_y_pixels[1:])**2)

        valid = np.where(d_pixel>1.0e-6)
        self._border_x_pixels = self._border_x_pixels[valid]
        self._border_y_pixels = self._border_y_pixels[valid]

        sorted_dex = np.argsort(self._border_x_pixels)
        self._border_x_pixels_by_x = self._border_x_pixels[sorted_dex]
        self._border_y_pixels_by_x = self._border_y_pixels[sorted_dex]

        # create lookups that allow you to lookup the x (y) values
        # that bound rows (columns)

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

    def _build_raw_boundary(self, resolution, threshold_factor):
        """
        Generate naive lists of x and y values for pixels on the boundary
        of the annotation (self._build_boundary will clean this list up
        for redundant pixels)
        """
        border_x = []
        border_y = []
        self._setup_drawing()
        n_segments = self.n_segments()
        self._border_interpolator = []
        for i1 in range(n_segments):
            if i1<n_segments-1:
                i2 = i1+1
            else:
                i2 = 0
            d_max = 10.0*resolution

            # sample each curve at a fine enough resolution
            # that we will get all of the border pixels
            tt = np.arange(0.0, 1.01, 0.1)
            d_threshold = threshold_factor*resolution
            while d_max>d_threshold:
                xx, yy = self._draw(i1, tt)
                dx = np.abs(xx[:-1]-xx[1:])
                dy = np.abs(yy[:-1]-yy[1:])
                dist = np.where(dx>dy,dx,dy)
                d_max = dist.max()
                if d_max>d_threshold:
                   bad_dex = np.where(dist>d_threshold)[0]
                   new_t = tt[bad_dex]+0.5*(tt[bad_dex+1]-tt[bad_dex])
                   tt = np.sort(np.concatenate([tt, new_t]))
            self._border_interpolator.append({'t':tt,
                                              'x':xx,
                                              'y':yy})
            border_x.append(xx)
            border_y.append(yy)
        border_x = np.concatenate(border_x)
        border_y = np.concatenate(border_y)
        return border_x, border_y


class SplineAnnotation(AnnotationBase):
    """
    An annotation in which boundaries are drawn as
    parametric curves in which x(t) and y(t) are
    cubic splines.
    """

    def n_segments(self):
        return len(self._spline.x)

    def _setup_drawing(self):
        self._spline = Spline2D(self._x_vals, self._y_vals)

    def _draw(self, i_segment, t_values):
        return self._spline.values(i_segment, t_values)


class PolyLineAnnotation(AnnotationBase):
    """
    An annotation in which boundaries are drawn as straight
    lines between points
    """

    def n_segments(self):
        return len(self._x_vals)

    def _setup_drawing(self):
        n_segments = self.n_segments()
        self._x_slopes = np.zeros(n_segments, dtype=float)
        self._x_intercepts = np.zeros(n_segments, dtype=float)
        self._y_slopes = np.zeros(n_segments, dtype=float)
        self._y_intercepts = np.zeros(n_segments, dtype=float)

        end_pts = []
        for ii in range(n_segments-1):
            end_pts.append((self._x_vals[ii], self._y_vals[ii],
                            self._x_vals[ii+1], self._y_vals[ii+1]))

        end_pts.append((self._x_vals[-1], self._y_vals[-1],
                        self._x_vals[0], self._y_vals[0]))

        for i_segment, (x0, y0, x1, y1) in enumerate(end_pts):
            self._x_slopes[i_segment] = x1-x0
            self._x_intercepts[i_segment] = x1-self._x_slopes[i_segment]
            self._y_slopes[i_segment] = y1-y0
            self._y_intercepts[i_segment] = y1-self._y_slopes[i_segment]

    def _draw(self, i_segment, t_values):
        xx = self._x_intercepts[i_segment] + t_values*self._x_slopes[i_segment]
        yy = self._y_intercepts[i_segment] + t_values*self._y_slopes[i_segment]
        return xx, yy


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
