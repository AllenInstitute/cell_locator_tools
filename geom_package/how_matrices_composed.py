import os
import re
import json
import numpy as np
import planar_geometry

def matrix_from_fname(fname):
    with open(fname, 'rb') as in_file:
        annotation_dict = json.load(in_file)
    data = annotation_dict['DefaultSplineOrientation']
    m = np.zeros((3,3), dtype=float)
    ii = 0
    for irow in range(3):
        for icol in range(3):
            m[irow,icol] = data[ii]
            ii+= 1
        ii += 1

    #c1 = m[:,1]
    #c2 = m[:,2]
    #m[:,1] = c2
    #m[:,2] = c1

    r1 = np.copy(m[1,:])
    r2 = np.copy(m[2,:])
    m[1,:] = r2
    m[2,:] = r1
    
    return m





if __name__ == "__main__":
    order_list = [('yaw', 'roll', 'pitch'),
                  ('yaw', 'pitch', 'roll'),
                  ('roll', 'yaw', 'pitch'),
                  ('roll', 'pitch', 'yaw'),
                  ('pitch', 'yaw', 'roll'),
                  ('pitch', 'roll', 'yaw')]


    json_dir = '../CellLocatorAnnotations'
    assert os.path.isdir(json_dir)
    composite_pattern = re.compile('annotation_y[0-9]*p[0-9]*r[0-9]*\.json')
    f_list = os.listdir(json_dir)
    good_composites = []
    no_go = []
    for fname in f_list:
        matches = composite_pattern.findall(fname)
        if len(matches)>0:
            good_composites.append(fname)

    y_pattern = re.compile('y[0-9]*')
    r_pattern = re.compile('r[0-9]*')
    p_pattern = re.compile('p[0-9]*')
    for composite_name in good_composites:
        p_string = composite_name.replace('annotation_','')
        params = {}

        yaw_s = y_pattern.search(p_string)
        params['yaw'] = p_string[yaw_s.start():yaw_s.end()]

        roll_s = r_pattern.search(p_string)
        params['roll'] = p_string[roll_s.start():roll_s.end()]

        pitch_s = p_pattern.search(p_string)
        params['pitch'] = p_string[pitch_s.start():pitch_s.end()]

        #print(fname,yaw,roll,pitch)
        is_valid = True
        matrices = {}
        for suffix in ('yaw', 'pitch', 'roll'):
            full_name = os.path.join(json_dir, 'annotation_%s.json' % params[suffix])
            if not os.path.isfile(full_name):
                #print('no %s' % full_name)
                is_valid = False
                break
            matrices[suffix] = matrix_from_fname(full_name)

        if not is_valid:
            continue

        #for suffix in ('yaw', 'pitch', 'roll'):
        if False:
            ang_deg = float(params[suffix][1:])
            print(suffix,params[suffix],ang_deg)
            print('matrix')
            print(matrices[suffix])
            pos_x = planar_geometry.rot_about_x(ang_deg)
            neg_x = planar_geometry.rot_about_x(-1*ang_deg)

            pos_y = planar_geometry.rot_about_y(ang_deg)
            neg_y = planar_geometry.rot_about_y(-1*ang_deg)

            pos_z = planar_geometry.rot_about_z(ang_deg)
            neg_z = planar_geometry.rot_about_z(-1*ang_deg)

            print('x')
            for ii in range(3):
               print(pos_x[ii,:],'    ',neg_x[ii,:])
            

            print('\ny')
            for ii in range(3):
               print(pos_y[ii,:],'    ',neg_y[ii,:])
            
            print('\nz')
            for ii in range(3):
               print(pos_z[ii,:],'    ',neg_z[ii,:])
            

        #continue

        composite_matrix = matrix_from_fname(os.path.join(json_dir, composite_name))
        eps = 1.0e-10
        found_match = False
        for order in order_list:
            test = np.dot(matrices[order[0]],
                          np.dot(matrices[order[1]],
                                 matrices[order[2]]))
            err_max = np.max(np.abs(test-composite_matrix))
            if err_max<=eps:
                print(composite_name,order)
                found_match = True
                break
            else:
                print(test)
        if not found_match:
            print('no order for ',composite_name)
            print(composite_matrix)

        print('testing individual rotations...')
        for suffix, mat_method in zip(('yaw', 'roll', 'pitch'),
                                      (planar_geometry.rot_about_y,
                                       planar_geometry.rot_about_x,
                                       planar_geometry.rot_about_z)):

            ang_deg = -1.0*float(params[suffix][1:])
            mtest = mat_method(ang_deg)
            d = np.max(np.abs(mtest-matrices[suffix]))
            assert d<=eps
            print(suffix)
