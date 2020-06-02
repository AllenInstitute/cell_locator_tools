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

class ImageGenerationTest(unittest.TestCase):

    def test_images(self):
        resolution = 25
        img_name = 'atlasVolume.mhd'
        img = SimpleITK.ReadImage(img_name)
        img_data = SimpleITK.GetArrayFromImage(img)
        brain_img = cell_locator_utils.BrainImage(img_data, resolution)
        print(img_data.shape)
        t0 = time.time()

        for dex in range(4):
            t1 = time.time()

            annotation_name = '../CellLocatorAnnotations/annotation_%d.json' % dex
            img_name = tempfile.mkstemp(dir='.',
                                    prefix='annotation_%d_' % dex,
                                    suffix='.png')[1]
            control_name = 'control_imgs/annotation_%d.png' % dex
            assert os.path.isfile(control_name)

            new_img = brain_img.slice_img_from_annotation(annotation_name)
            plt.figure(figsize=(10,10))
            plt.imshow(new_img)
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

if __name__ == "__main__":
    unittest.main()
