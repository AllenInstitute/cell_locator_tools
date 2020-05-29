import numpy as np


class Spline2D(object):

    def __init__(self, x_vals, y_vals):
        self._x = self._get_coefficients(x_vals)
        self._y = self._get_coefficients(y_vals)

    def values(self, ii, t):
        t2 = t**2
        t3 = t**3
        v_x = self._x[ii, 0] + self._x[ii, 1]*t + self._x[ii, 2]*t2 + self._x[ii, 3]*t3
        v_y = self._y[ii, 0] + self._y[ii, 1]*t + self._y[ii, 2]*t2 + self._y[ii, 3]*t3
        return (v_x, v_y)

    def derivatives(self, ii, t):
        t2 = t**2
        vp_x = self._x[ii, 1] + 2*self._x[ii, 2]*t + 3*self._x[ii, 3]*t2
        vp_y = self._y[ii, 1] + 2*self._y[ii, 2]*t + 3*self._y[ii, 3]*t2
        return (vp_x, vp_y)

    def second_derivatives(self, ii, t):
        vpp_x = 2*self._x[ii, 2] + 6*self._x[ii, 3]*t
        vpp_y = 2*self._y[ii, 2] + 6*self._y[ii, 3]*t
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
