import numpy as np
import os
import SimpleITK
import json
from voxel_mask import VoxelMask
import copy
import time
import argparse


if __name__ == "__main__":
    fname = '../marga_json_files_20200619/1A.json'
    fname = '../CellLocatorAnnotations/annotation_20200602.json'
    assert os.path.isfile(fname)

    resolution  = 10
    img_name = 'average_template_10.nrrd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)
    img_shape = copy.deepcopy(img_data.shape)

    del img_data

    ann_dir = '../marga_json_files_20200619'
    fname_list = os.listdir(ann_dir)
    ct = 0

    voxel_mask = VoxelMask(img_shape[2], img_shape[1], img_shape[0], resolution)

    t0 = time.time()
    for n in fname_list[:10]:
        if not n.endswith('json'):
            continue
        fname = os.path.join(ann_dir, n)

        with open(fname, 'rb') as in_file:
            annotation = json.load(in_file)
            markup = annotation['Markups'][0]

        print('go')
        test = voxel_mask.get_voxel_mask(markup)

        ct += 1
    dur = time.time()-t0
    print('ran ',ct)
    per = dur/ct
    print('mask per %e seconds' % per)
