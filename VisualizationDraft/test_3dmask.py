import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import os
import SimpleITK
import json
import cell_locator_utils
import copy
import time

if __name__ == "__main__":
    resolution = 25
    img_name = 'atlasVolume.mhd'
    img = SimpleITK.ReadImage(img_name)
    img_data = SimpleITK.GetArrayFromImage(img)

    brain_vol = cell_locator_utils.BrainVolume(img_data, resolution)
    dummy_brain_vol = copy.deepcopy(brain_vol)


    annotation_name = '../CellLocatorAnnotations/annotation_20200602.json'
    assert os.path.isfile(annotation_name)

    slice_img = brain_vol.slice_img_from_annotation(annotation_name,
                                                    from_pts=True)

    with open(annotation_name, 'rb') as in_file:
        annotation_dict = json.load(in_file)
    markup = annotation_dict['Markups'][0]
    t0 = time.time()
    annotation = slice_img.brain_slice.annotation_from_markup(markup)
    pixel_mask = annotation.get_mask(25)
    print('returned mask in %e' % (time.time()-t0))

    (pixel_coords,
            valid) = slice_img.brain_slice.allen_to_pixel(brain_vol.brain_volume)
    print('pixel coords %e' % (time.time()-t0))
    old_valid = valid

    t1 = time.time()
    max_x = np.nanmax(pixel_coords[0,:]).astype(int)+1
    max_y = np.nanmax(pixel_coords[1,:]).astype(int)+1
    print('nanmax %e' % (time.time()-t1))

    good_pixels = np.where(pixel_mask)
    print('np.where in %e' % (time.time()-t0))
    good_pixel_indices = good_pixels[1]*max_y + good_pixels[0]
    print('good pixels in %e' % (time.time()-t0))

    test_pixel_indices = pixel_coords[0,:]*max_y+pixel_coords[1,:]
    print('test_pixels in %e' % (time.time()-t0))

    valid_voxels = np.isin(test_pixel_indices, good_pixel_indices)
    print('got valid_voxels in %e seconds' % (time.time()-t0))

    #print('valid voxels %d' % valid_voxels.sum())
    #print(test_pixel_indices.dtype)
    #print(good_pixel_indices.dtype)
    #print(good_pixel_indices[:10])
    #print(test_pixel_indices[:10])

    #print(pixel_coords.shape)
    #print(brain_vol.brain_volume.shape)

    for ix in range(2):
        assert np.isnan(pixel_coords[ix,:][valid_voxels]).sum() == 0
        #print('pix %d '% ix, pixel_coords[ix,:])

    dummy_brain_vol.img_data = dummy_brain_vol.img_data.astype(float)

    val = dummy_brain_vol.img_data.max()
    s0 = dummy_brain_vol.img_data.sum()
    dummy_brain_vol.img_data[valid_voxels] += 2*val
    s1 = dummy_brain_vol.img_data.sum()
    #print('difference of sums ',(s1-s0)/val,val,dummy_brain_vol.img_data.dtype)
    #print('valid_voxel shape ',valid_voxels.shape, valid_voxels.sum())
    #print(dummy_brain_vol.img_data.shape)
    dummy_slice_img = dummy_brain_vol.slice_img_from_annotation(annotation_name,
                                                    from_pts=True)

    assert not np.array_equal(slice_img.img, dummy_slice_img.img)
    #print('slice ',slice_img.img.sum())
    #print('dummy ',dummy_slice_img.img.sum())
    #print('diff ',(slice_img.img.sum()-dummy_slice_img.img.sum())/val)

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(slice_img.img)
    plt.subplot(1,2,2)
    plt.imshow(dummy_slice_img.img)
    plt.savefig('use_voxel_mask.pdf')
    exit()

    ct = 0
    ct_in_plane = 0
    voxel_set = set(np.where(valid_voxels)[0])
    for ii in dummy_brain_vol._save_img_dex_flat:
        if ii in voxel_set:
            ct += 1
        if np.logical_not(np.isnan(pixel_coords[0,ii])) and np.logical_not(np.isnan(pixel_coords[1,ii])):
            ct_in_plane += 1


    np.testing.assert_array_equal(old_valid, dummy_slice_img.brain_slice._init_valid_dex)

    print('ct ',ct)
    print('ct in plane ',ct_in_plane)
    print('saved ',dummy_brain_vol._save_img_dex_flat.shape)
    print(old_valid.shape)

    img_pixels = brain_vol.brain_volume[:,dummy_brain_vol._save_img_dex_flat]
    s = dummy_slice_img.brain_slice.coord_converter.allen_to_slice(img_pixels)
    print(s)
