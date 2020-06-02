import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import SimpleITK
import cell_locator_utils

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None)
    parser.add_argument('--pts', action='store_true', default=False)
    args = parser.parse_args()

    if args.annotation is None:
        raise RuntimeError("must specify annotation")

    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)

    brain_img = cell_locator_utils.BrainImage(img_data, resolution)
    (slice_img,
     brain_slice) = brain_img.slice_img_from_annotation(args.annotation,
                                                        from_pts=args.pts)

    plt.figure(figsize=(15,15))
    plt.imshow(slice_img)
    plt.show()
