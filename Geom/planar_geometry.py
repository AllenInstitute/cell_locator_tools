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
        return cls.plane_from_points(planar_pts[0],
                                     planar_pts[1],
                                     planar_pts[2])
        
