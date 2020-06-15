import numpy as np

def v_from_pts(pt1, pt2):
    v = pt1-pt2
    n = np.sqrt(np.sum(v**2))
    if n != 0.0:
        return v/n
    return v

def v_cross(v1, v2):
    v = np.array([v1[1]*v2[2]-v1[2]*v2[1],
                  v1[2]*v2[0]-v1[0]*v2[2],
                  v1[0]*v2[1]-v1[1]*v2[0]])
    return v/np.sqrt(np.sum(v**2))

def rot_about_z(ang_deg):
    theta = np.radians(ang_deg)
    m = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    return m

def rot_about_y(ang_deg):
    theta = np.radians(ang_deg)
    m = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(theta), 0.0, np.cos(theta)]])
    return m

def rot_about_x(ang_deg):
    theta = np.radians(ang_deg)
    m = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(theta), -np.sin(theta)],
                  [0.0, np.sin(theta), np.cos(theta)]])

    return m


def rotate_v_into_w_2d(v, w, already_normed=False):
    """
    Find matrix that rotates the vector v into the vector w
    """
    if not already_normed:
        v = v/np.sqrt(np.sum(v**2))
        w = w/np.sqrt(np.sum(w**2))
    aa = w[0]*v[0]+w[1]*v[1]
    bb = w[0]*v[1]-w[1]*v[0]
    return np.array([[aa, bb],[-bb,aa]])


def rotate_v_into_w_3d(v, w):
    # first find matrices to rotate v, w in to the y,z plane
    # then find the matrix to rotate those projections into each other
    # then apply the inverse of the matrix rotating w in the y,z plane

    v = v/np.sqrt(np.sum(v**2))
    w = w/np.sqrt(np.sum(w**2))
    v_abs = np.abs(v)
    w_abs = np.abs(w)
    v2d_to_yz = np.identity(3, dtype=float)
    if v_abs[:2].max()>0.0:
        v2d_to_yz[:2,:2] = rotate_v_into_w_2d(v[:2], np.array([0.0,1.0]),
                                              already_normed=False)

    w2d_to_yz = np.identity(3, dtype=float)
    if w_abs[:2].max()>0.0:
        w2d_to_yz[:2, :2] = rotate_v_into_w_2d(w[:2], np.array([0.0, 1.0]),
                                               already_normed=False)

    tmp_m = rotate_v_into_w_2d(np.dot(v2d_to_yz, v)[1:],
                               np.dot(w2d_to_yz, w)[1:])

    v_to_w_yz = np.identity(3, dtype=float)
    v_to_w_yz[1:,1:] = tmp_m

    return np.dot(w2d_to_yz.transpose(),
                  np.dot(v_to_w_yz, v2d_to_yz))


class Plane(object):

    @property
    def origin(self):
        return np.copy(self._origin)

    @property
    def normal(self):
        return np.copy(self._normal)

    @property
    def eps(self):
        return self._eps

    def __init__(self, origin, normal):
        self._origin = np.copy(origin)
        self._normal = np.copy(normal)
        self._normal /= np.sqrt(np.sum(self._normal**2))
        self._eps = 1.0e-10

    def set_origin(self, origin_in):
        if not self.in_plane(origin_in):
            raise RuntimeError("Cannot set origin; this point not in plane")
        self._origin = np.copy(origin_in)

    def in_plane(self, pt, tol=None):
        if tol is not None:
            local_eps = tol
        else:
            local_eps = self.eps
        v = v_from_pts(pt, self.origin)
        d = np.abs(np.dot(v, self.normal))
        if d<local_eps:
            return True
        print('not in plane: %e (tol %e)' % (d, local_eps))
        return False

    @classmethod
    def plane_from_points(cls, pt1, pt2, pt3):
        origin = pt3
        v1 = v_from_pts(pt1, origin)
        v2 = v_from_pts(pt2, origin)
        norm = v_cross(v1, v2)
        return cls(origin, norm)

    @classmethod
    def plane_from_many_points(cls, pt_list):
        planar_pts = [pt_list[0]]
        for ii in range(1,len(pt_list),1):
            is_valid = True
            for jj in range(len(planar_pts)):
                v = v_from_pts(pt_list[ii], planar_pts[jj])
                if np.abs(np.sum(v**2))<1.0e-10:
                    is_valid = False
                    break
            if is_valid:
                planar_pts.append(pt_list[ii])
                if len(planar_pts) == 3:
                    break
        if len(planar_pts)!=3:
            raise RuntimeError("Not enough indpendent points")
        p = cls.plane_from_points(planar_pts[0],
                                  planar_pts[1],
                                  planar_pts[2])

        d_sum = 0.0
        d_ct = 0.0
        for i1 in range(len(pt_list)):
            for i2 in range(i1+1, len(pt_list), 1):
                d = np.sqrt(np.sum(pt_list[i1]-pt_list[i2])**2)
                d_sum += d
                d_ct += 1.0
        d_avg = d_sum/d_ct

        for pt in pt_list:
            #print('')
            #print(pt_list)
            #print(planar_pts)
            #print(d_avg)
            assert p.in_plane(pt, tol=1.0e-2*d_avg)
        return p
