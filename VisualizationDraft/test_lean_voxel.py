import numpy as np
import os
import SimpleITK
import json
import cell_locator_utils
from lean_voxel import VoxelMask
import copy
import time
import argparse

def get_control_mask(annotation_fname, brain_vol, img_shape):

    with open(annotation_fname, 'rb') as in_file:
        annotation_dict = json.load(in_file)
    markup = annotation_dict['Markups'][0]
    brain_slice = brain_vol.brain_slice_from_annotation(annotation_fname)
    valid_voxels = brain_vol.get_voxel_mask(brain_slice, markup)
    return valid_voxels


if __name__ == "__main__":
    fname = '../marga_json_files_20200619/1A.json'
    fname = '../CellLocatorAnnotations/annotation_20200602.json'
    assert os.path.isfile(fname)

    resolution  = 10
    img_name = 'average_template_10.nrrd'

    resolution = 25
    img_name = 'atlasVolume.mhd'

    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)
    img_shape = copy.deepcopy(img_data.shape)
    brain_vol = cell_locator_utils.BrainVolume(img_data, resolution, keep_img_data=False)

    del img_data

    ann_dir = '../marga_json_files_20200619'
    fname_list = os.listdir(ann_dir)
    fname_list.sort()

    ann_dir = '../CellLocatorAnnotations'
    fname_list = ['annotation_20200602.json']
    ct = 0

    voxel_mask = VoxelMask(img_shape[2], img_shape[1], img_shape[0], resolution)

    for n in fname_list[:4]:
        if not n.endswith('json'):
            continue
        fname = os.path.join(ann_dir, n)

        control = get_control_mask(fname, brain_vol, img_shape)
        print('got control')

        with open(fname, 'rb') as in_file:
            annotation = json.load(in_file)
            markup = annotation['Markups'][0]

        test = voxel_mask.get_voxel_mask(markup)

        print('control %e' % control.sum())
        print('test %e' % test.sum())
        test_not_control = np.logical_and(test, np.logical_not(control)).sum()
        control_not_test = np.logical_and(control, np.logical_not(test)).sum()
        print('test_not_control ',test_not_control)
        print('control_not_test ',control_not_test)
        np.testing.assert_array_equal(control, test)
        ct += 1
    print('ran ',ct)
