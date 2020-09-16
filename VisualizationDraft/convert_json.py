import SimpleITK
from lean_voxel import VoxelMask
import json
import numpy as np
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas_name', type=str, default='atlasVolume.mhd')
    parser.add_argument('--json_name', type=str, default=None)
    parser.add_argument('--out_name', type=str, default=None)
    args = parser.parse_args()

    if args.out_name is None:
        raise RuntimeError('Must specify out_name')
    if args.json_name is None:
        raise RuntimeError('Must specify json_name')

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
                          dtype='uint8')

    json_file_list = []
    if os.path.isfile(args.json_name):
        json_file_list.append(args.json_name)
    elif os.path.isdir(args.json_name):
        file_name_list = os.listdir(args.json_name)
        for name in file_name_list:
            if name.endswith('json'):
                json_file_list.append(os.path.join(args.json_name, name))


    for ii, json_name in enumerate(json_file_list):

        with open(json_name, 'rb') as in_file:
            annotation = json.load(in_file)
            markup = annotation['Markups'][0]

        mask = voxel_mask.get_voxel_mask(markup)
        output_img[mask] = ii+1



    output_img = SimpleITK.GetImageFromArray(output_img.reshape(atlas_array.shape))
    output_img.SetSpacing((resolution, resolution, resolution))
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(args.out_name)
    writer.SetUseCompression(True)
    writer.Execute(output_img)
