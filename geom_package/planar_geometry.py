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


def rotate_v_into_w_2d(v, w):
    """
    Find matrix that rotates the vector v into the vector w
    """
    v_n = v/np.sqrt(np.sum(v**2))
    w_n = w/np.sqrt(np.sum(w**2))
    aa = w_n[0]*v_n[0]+w_n[1]*v_n[1]
    bb = w_n[0]*v_n[1]-w_n[1]*v_n[0]
    return np.array([[aa, bb],[-bb,aa]])


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

    def in_plane(self, pt):
        v = v_from_pts(pt, self.origin)
        d = np.abs(np.dot(v, self.normal))
        if d<self.eps:
            return True
        print('not in plane: %e' % d)
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
        for pt in pt_list:
            assert p.in_plane(pt)
        return p
