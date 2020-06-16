import numpy as np
import os
import SimpleITK
import json
import cell_locator_utils
import copy
import time
import multiprocessing

import argparse

def validate_markup(markup):
    if markup is None:
        return False
    if len(markup['Points'])<3:
        return False
    return True

def write_annotation(annotation_fname_list, annotation_dir, brain_vol, out_dir):
    label = 1
    t0 = time.time()
    for i_file, fname in enumerate(annotation_fname_list):
        if not fname.endswith('json'):
            continue
        annotation_name = os.path.join(annotation_dir, fname)

        with open(annotation_name, 'rb') as in_file:
            annotation_dict = json.load(in_file)
        markup = annotation_dict['Markups'][0]
        is_valid= validate_markup(markup)
        if not is_valid:
            continue

        brain_slice = brain_vol.brain_slice_from_annotation(annotation_name)

        output_voxels = np.zeros(brain_vol.brain_volume.shape[1], dtype=np.uint16)

        valid_voxels = brain_vol.get_voxel_mask(brain_slice, markup)

        output_voxels[valid_voxels] = label

        output_voxels = output_voxels.reshape(img_shape)
        output_img = SimpleITK.GetImageFromArray(output_voxels)
        output_img.SetSpacing((resolution, resolution, resolution))
        writer = SimpleITK.ImageFileWriter()
        out_name = fname.replace('.json', '.nrrd')
        writer.SetFileName(os.path.join(out_dir, out_name))
        writer.Execute(output_img)
        if i_file>0 and i_file%10 == 0:
            duration = (time.time()-t0)/3600.0
            per = duration/i_file
            pred = per*len(annotation_fname_list)
            print('ran on %d in %.2d hrs; expect %.2e hrs' %
            (i_file, duration, pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, default='average_template_10.nrrd')
    parser.add_argument('--in_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--n_threads', type=int, default=2)
    args = parser.parse_args()

    assert os.path.isfile(args.template)
    assert os.path.isdir(args.in_dir)
    assert os.path.isdir(args.out_dir)

    resolution  = 10
    img_name = args.template
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)
    img_shape = copy.deepcopy(img_data.shape)
    brain_vol = cell_locator_utils.BrainVolume(img_data, resolution, keep_img_data=False)

    del img_data
    del img

    annotation_dir = args.in_dir
    annotation_fname_list = os.listdir(annotation_dir)
    annotation_fname_list.sort()

    annotation_fname_list = annotation_fname_list[:5]

    p_list = []
    per_thread = len(annotation_fname_list)//args.n_threads

    for i0 in range(0, len(annotation_fname_list), per_thread):
        sub_list = annotation_fname_list[i0:i0+per_thread]
        p = multiprocessing.Process(target=write_annotation,
                                    args=(sub_list, annotation_dir, brain_vol, args.out_dir))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    print('done')
