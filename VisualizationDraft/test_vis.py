import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import SimpleITK

import json
import time
import tempfile
import hashlib

import unittest
import cell_locator_utils

class BrainSliceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fname = '../CellLocatorAnnotations/annotation_unittest.json'
        assert os.path.isfile(fname)
        with open(fname, 'rb') as in_file:
            cls.full_annotation = json.load(in_file)
        resolution = 25
        img_name = 'atlasVolume.mhd'
        img = SimpleITK.ReadImage(img_name)
        img_data = SimpleITK.GetArrayFromImage(img)
        cls.brain_vol = cell_locator_utils.BrainVolume(img_data, resolution)

    def test_pixel_to_slice(self):
        coord_converter = cell_locator_utils.CellLocatorTransformation(self.full_annotation)
        brain_slice = cell_locator_utils.BrainSlice(coord_converter,
                                                    self.brain_vol.resolution,
                                                    self.brain_vol.brain_volume)
        rng = np.random.RandomState(8123512)
        pixel0 = rng.randint(0,4000, size=(2,5000))
        slice_coords = brain_slice.pixel_to_slice(pixel0)
        pixel1 = brain_slice.slice_to_pixel(slice_coords)
        np.testing.assert_array_equal(pixel0, pixel1)


class ImageGenerationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resolution = 25
        img_name = 'atlasVolume.mhd'
        img = SimpleITK.ReadImage(img_name)
        img_data = SimpleITK.GetArrayFromImage(img)
        cls.brain_vol = cell_locator_utils.BrainVolume(img_data, cls.resolution)

    def test_images(self):

        t0 = time.time()

        for dex in range(4):
            t1 = time.time()

            annotation_name = '../CellLocatorAnnotations/annotation_%d.json' % dex
            img_name = tempfile.mkstemp(dir='.',
                                    prefix='annotation_%d_' % dex,
                                    suffix='.png')[1]
            control_name = 'control_imgs/annotation_%d.png' % dex
            assert os.path.isfile(control_name)

            slice_img = self.brain_vol.slice_img_from_annotation(annotation_name)

            plt.figure(figsize=(10,10))
            plt.imshow(slice_img.img)
            plt.savefig(img_name)
            print(img_name)

            md5_control = hashlib.md5()
            with open(control_name, 'rb') as in_file:
                while True:
                    data = in_file.read(10000)
                    if len(data)==0:
                        break
                    md5_control.update(data)
            md5_test = hashlib.md5()
            with open(img_name, 'rb') as in_file:
                while True:
                    data = in_file.read(10000)
                    if len(data)==0:
                        break
                    md5_test.update(data)
            self.assertEqual(md5_test.hexdigest(), md5_control.hexdigest())

            print(md5_test.hexdigest())
            print(md5_control.hexdigest())
            print('')
            if os.path.exists(img_name):
                os.unlink(img_name)

    def test_from_pts(self):
        """
        Test that we get the same mask when plane is instantiated from points,
        rather than just from the transformation matrix
        """

        annotation_fname = '../CellLocatorAnnotations/annotation_unittest.json'
        self.assertTrue(os.path.isfile(annotation_fname))

        with open(annotation_fname, 'rb') as in_file:
            full_annotation = json.load(in_file)
        c1 = cell_locator_utils.CellLocatorTransformation(full_annotation)
        c2 = cell_locator_utils.CellLocatorTransformation(full_annotation['Markups'][0],
                                                          from_pts=True)

        mask1 = c1.get_slice_mask_from_allen(self.brain_vol.brain_volume,
                                             self.brain_vol.resolution)
        mask2 = c2.get_slice_mask_from_allen(self.brain_vol.brain_volume,
                                             self.brain_vol.resolution)

        np.testing.assert_equal(mask1, mask2)

        valid = np.where(mask1)
        s_coords1 = c1.allen_to_slice(self.brain_vol.brain_volume[:,valid[0]])
        s_coords2 = c2.allen_to_slice(self.brain_vol.brain_volume[:,valid[0]])
        self.assertLess(np.abs(s_coords1[2,:]).max(), 0.5*self.brain_vol.resolution)
        self.assertLess(np.abs(s_coords2[2,:]).max(), 0.5*self.brain_vol.resolution)


if __name__ == "__main__":
    unittest.main()
