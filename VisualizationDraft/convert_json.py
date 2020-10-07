import SimpleITK
from voxel_mask import VoxelMask
import json
import numpy as np
import os
import argparse
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas_name', type=str, default='atlasVolume.mhd',
                        help='the mhd volume defining the mouse brain atlas'
                        ' volume, such as can be downloaded here: '
                        'http://help.brain-map.org/display/mousebrain/API#API-DownloadAtlas')
    parser.add_argument('--json_name', type=str, default=None,
                        help='the name of the json file (or a directory'
                        ' containing json files) output by CellLocator'
                        ' which you would like to convert')
    parser.add_argument('--out_name', type=str, default=None,
                        help='the name of the nrrd file you would like to'
                        ' produce (Note: if json_name is the name of a dir,'
                        ' the nrrd file will contain all of the annotations in'
                        ' that dir, each with a unique color-specifying int)')
    args = parser.parse_args()

    if args.out_name is None:
        raise RuntimeError('Must specify out_name')
    if args.json_name is None:
        raise RuntimeError('Must specify json_name')

    if not os.path.isfile(args.atlas_name):
        raise RuntimeError("Atlas file\n%s\ndoes not exist" % args.atlas_name)

    with open(args.atlas_name, 'r') as in_file:
        for line in in_file:
            params = line.strip().split()
            if params[0] == 'ElementSpacing':
                resolution = int(params[2])

    print('resolution ',resolution)

    atlas_img = SimpleITK.ReadImage(args.atlas_name)

    atlas_array = atlas_img = SimpleITK.GetArrayFromImage(atlas_img)

    voxel_mask = VoxelMask(atlas_array.shape[2],
                           atlas_array.shape[1],
                           atlas_array.shape[0],
                           resolution)

    output_img = np.zeros(atlas_array.shape[0]*atlas_array.shape[1]*atlas_array.shape[2],
                          dtype=int)

    json_file_list = []
    if os.path.isfile(args.json_name):
        json_file_list.append(args.json_name)
    elif os.path.isdir(args.json_name):
        file_name_list = os.listdir(args.json_name)
        for name in file_name_list:
            if name.endswith('json'):
                json_file_list.append(os.path.join(args.json_name, name))


    t0 = time.time()
    for ii, json_name in enumerate(json_file_list):

        with open(json_name, 'rb') as in_file:
            annotation = json.load(in_file)
            markup = annotation['Markups'][0]
        if markup is None:
            continue

        mask = voxel_mask.get_voxel_mask(markup)
        output_img[mask] = ii+1
        if ii%100 == 0 and ii>0:
            duration = time.time()-t0
            per = duration/ii
            pred = per*len(json_file_list)
            print('%d in %e seconds (per %e; predict %e)' % (ii,duration,per,pred))

    print('took %e s per' % ((time.time()-t0)/len(json_file_list)))



    output_img = SimpleITK.GetImageFromArray(output_img.reshape(atlas_array.shape))
    output_img.SetSpacing((resolution, resolution, resolution))
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(args.out_name)
    writer.SetUseCompression(True)
    writer.Execute(output_img)
