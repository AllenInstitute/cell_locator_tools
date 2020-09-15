import SimpleITK
from lean_voxel import VoxelMask
import json
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

    with open(args.json_name, 'rb') as in_file:
        annotation = json.load(in_file)
        markup = annotation['Markups'][0]

    mask = voxel_mask.get_voxel_mask(markup)
    print(mask)
    print(mask.shape)

    output_img = SimpleITK.GetImageFromArray(mask.reshape(atlas_array.shape).astype('uint8'))
    output_img.SetSpacing((resolution, resolution, resolution))
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(args.out_name)
    writer.SetUseCompression(True)
    writer.Execute(output_img)
