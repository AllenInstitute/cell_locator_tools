import json
import os
import numpy as np
import planar_geometry
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_name', type=str, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.in_name)

    with open(args.in_name, 'rb') as in_file:
        annotation = json.load(in_file)
    markup = annotation['Markups'][0]
    
    pt_list = []
    for obj in markup['Points']:
        pt = np.array([obj['x'], obj['y'], obj['z']])
        pt_list.append(pt)
    plane = planar_geometry.Plane.plane_from_many_points(pt_list)
    for pt in pt_list:
        assert plane.in_plane(pt)

    print('%d pts' % (len(pt_list)))
    #for pt in pt_list:
    #    print(pt)

    spline_orientation = markup['SplineOrientation']
    transform = np.zeros((4,4), dtype=float)
    ii = 0
    for irow in range(4):
        for icol in range(4):
            transform[irow,icol] = spline_orientation[ii]
            ii += 1
    
    inverse_transform = np.linalg.inv(transform)
    
    rotation_matrix = np.copy(transform[:3,:3])
    #r1 = np.copy(rotation_matrix[1,:])
    #r2 = np.copy(rotation_matrix[2,:])
    #rotation_matrix[1,:] = r2
    #rotation_matrix[2,:] = r1

    x = np.array([1.0,0.0,0.0])
    y = np.array([0.0,1.0,0.0])
    z = np.array([0.0,0.0,1.0])
    print('normal')
    print(plane.normal)
    print('x')
    print(np.dot(rotation_matrix, x))
    print('y')
    print(np.dot(rotation_matrix, y))
    print('z')
    print(np.dot(rotation_matrix, z))
    new_z = np.dot(rotation_matrix, z)
    x_cross_y = planar_geometry.v_cross(np.dot(rotation_matrix, x),
                                        np.dot(rotation_matrix, y))

    # it looks like the 4x4 transformation matrix converts
    # coordinates in the plane into 3D coordinates
    # That is going to be more annoying

    rng = np.random.RandomState(88123)
    for ii in range(100):
        p = np.array([10000.0*rng.random_sample()-5000.0, 
                      10000.0*rng.random_sample()-5000.0, 0.0, 1.0])
        new_pt = np.dot(transform, p)[:3]
        assert plane.in_plane(new_pt)

    # so: inverse_transform maps 3D coordinates into
    # coordinates on the plane
    print('\napplying inverse transform to points')
    for pt in pt_list:
        print(np.dot(inverse_transform, np.append(pt,1.0)))

    exit()
    inverse_rotation = np.linalg.inv(rotation_matrix)

    p = np.array([5.0, 2.1, 0.0])
    new_pt = np.dot(rotation_matrix, p) + plane.origin
    print(plane.in_plane(new_pt))
    print(np.dot(inverse_rotation, plane.normal))

    t = np.copy(transform[:3,3])
    print('is t in plane ',plane.in_plane(t))

    origin = -1.0*np.dot(inverse_rotation, t)

    print('t ',t)
    print('o ',origin)
    print(plane.in_plane(origin))
    print(plane.origin)
    print(planar_geometry.v_from_pts(origin, plane.origin))
    print(plane.normal)
