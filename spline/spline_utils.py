import numpy as np


class Spline2D(object):

    def __init__(self, ordered_vals):
        mat, bc = self._construct_matrix(ordered_vals)


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
    values = np.array([1,2,3])
    s = Spline2D(values)
    m, b = s._construct_matrix(values)
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
