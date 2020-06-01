import unittest
import numpy as np
import planar_geometry

class TestVector(unittest.TestCase):

    def test_cross(self):
        v1 = np.array([1,2.4,3.2])
        v2 = np.array([5.2,8.9,-2.9])
        v3 = planar_geometry.v_cross(v1, v2)
        self.assertLess(np.abs(np.dot(v3,v1)), 1.0e-10)
        self.assertLess(np.abs(np.dot(v3,v2)), 1.0e-10)
        self.assertLess(np.abs(1.0-np.sum(v3**2)), 1.0e-10)

    def test_cross_xy(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(z,planar_geometry.v_cross(x,y))
        np.testing.assert_array_equal(-1.0*z,planar_geometry.v_cross(y,x))
        np.testing.assert_array_equal(-1.0*x,planar_geometry.v_cross(z,y))
        np.testing.assert_array_equal(x,planar_geometry.v_cross(y,z))
        np.testing.assert_array_equal(-1.0*y,planar_geometry.v_cross(x,z))
        np.testing.assert_array_equal(y,planar_geometry.v_cross(z,x))
        np.testing.assert_array_equal(-1.0*z,
                    planar_geometry.v_cross(x+y,x-y))

    def test_rot_x(self):
        ang = 42.0
        m = planar_geometry.rot_about_x(ang)
        rng = np.random.RandomState(22)
        v1 = rng.random_sample(3)
        v2 = np.dot(m,v1)
        self.assertEqual(v1[0], v2[0])
        self.assertLess(np.abs(np.sum(v1**2)-np.sum(v2**2)),
                        1.0e-12*np.sum(v1**2))

        v1_2d = np.array([v1[1],v1[2]])
        v2_2d = np.array([v2[1], v2[2]])
        d = np.dot(v1_2d, v2_2d)/np.sqrt(np.sum(v1_2d**2)*np.sum(v2_2d**2))
        self.assertLess(np.abs(d-np.cos(np.radians(ang))), 1.0e-10)

        v1 = np.array([0.0, 1.0, 1.0])
        v2 = np.dot(m, v1)
        self.assertGreater(v2[1], 0.0)
        self.assertGreater(v2[2], 0.0)
        self.assertEqual(v2[0], 0.0)
        self.assertEqual(v2[1], np.cos(np.radians(ang))-np.sin(np.radians(ang)))
        self.assertEqual(v2[2], np.sin(np.radians(ang))+np.cos(np.radians(ang)))

    def test_rot_y(self):
        ang = 42.0
        m = planar_geometry.rot_about_y(ang)
        rng = np.random.RandomState(832)
        v1 = rng.random_sample(3)
        v2 = np.dot(m,v1)
        self.assertEqual(v1[1], v2[1])
        self.assertLess(np.abs(np.sum(v1**2)-np.sum(v2**2)),
                        1.0e-12*np.sum(v1**2))

        v1_2d = np.array([v1[0],v1[2]])
        v2_2d = np.array([v2[0], v2[2]])
        d = np.dot(v1_2d, v2_2d)/np.sqrt(np.sum(v1_2d**2)*np.sum(v2_2d**2))
        self.assertLess(np.abs(d-np.cos(np.radians(ang))), 1.0e-10)

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.dot(m, v1)
        self.assertGreater(v2[0], 0.0)
        self.assertLess(v2[2], 0.0)
        self.assertEqual(v2[1], 0.0)
        self.assertEqual(v2[0], np.cos(np.radians(ang)))
        self.assertEqual(v2[2], -np.sin(np.radians(ang)))

        v1 = np.array([0.0, 0.0, 1.0])
        v2 = np.dot(m, v1)
        self.assertGreater(v2[0], 0.0)
        self.assertGreater(v2[2], 0.0)
        self.assertEqual(v2[1], 0.0)
        self.assertEqual(v2[0], np.sin(np.radians(ang)))
        self.assertEqual(v2[2], np.cos(np.radians(ang)))

    def test_rot_z(self):
        ang = 42.0
        m = planar_geometry.rot_about_z(ang)
        rng = np.random.RandomState(8912)
        v1 = rng.random_sample(3)
        v2 = np.dot(m,v1)
        self.assertEqual(v1[2], v2[2])
        self.assertLess(np.abs(np.sum(v1**2)-np.sum(v2**2)),
                        1.0e-12*np.sum(v1**2))

        v1_2d = np.array([v1[0],v1[1]])
        v2_2d = np.array([v2[0], v2[1]])
        d = np.dot(v1_2d, v2_2d)/np.sqrt(np.sum(v1_2d**2)*np.sum(v2_2d**2))
        self.assertLess(np.abs(d-np.cos(np.radians(ang))), 1.0e-10)

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.dot(m, v1)
        self.assertGreater(v2[0], 0.0)
        self.assertGreater(v2[1], 0.0)
        self.assertEqual(v2[2], 0.0)
        self.assertEqual(v2[0], np.cos(np.radians(ang)))
        self.assertEqual(v2[1], np.sin(np.radians(ang)))

        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.dot(m, v1)
        self.assertLess(v2[0], 0.0)
        self.assertGreater(v2[1], 0.0)
        self.assertEqual(v2[2], 0.0)
        self.assertEqual(v2[0], -np.sin(np.radians(ang)))
        self.assertEqual(v2[1], np.cos(np.radians(ang)))

    def test_rotating_2d_vectors(self):
        vv = np.array([0.0, 1.0])
        ww = np.array([-0.2, np.sqrt(1.0-0.04)])
        mm = planar_geometry.rotate_v_into_w_2d(vv, ww)
        test = np.dot(mm, vv)
        np.testing.assert_allclose(test, ww,
                                   atol=1.0e-10, rtol=1.0e-10)

        vv = np.array([1.0, 0.0])
        ww = np.array([0.72, -np.sqrt(1.0-0.72**2)])
        ww /= np.sqrt(np.sum(ww**2))
        self.assertGreater(np.abs(1.0-np.dot(vv,ww)), 0.1)
        mm = planar_geometry.rotate_v_into_w_2d(vv, ww)
        test = np.dot(mm, vv)
        np.testing.assert_allclose(test, ww,
                                   atol=1.0e-10, rtol=1.0e-10)

        rng = np.random.RandomState(8812)
        for ii in range(100):
            vv = 0.5-rng.random_sample(2)
            ww = 0.5-rng.random_sample(2)
            mm = planar_geometry.rotate_v_into_w_2d(vv, ww)
            test = np.dot(mm, vv)
            d = np.dot(test, ww)
            self.assertAlmostEqual(d, np.sqrt(np.sum(vv**2)*np.sum(ww**2)),
                                   delta=np.abs(d)*1.0e-10)

        for ii in range(100):
            vv = 0.5-rng.random_sample(2)
            ww = 0.5-rng.random_sample(2)
            vv /= np.sqrt(np.sum(vv**2))
            ww /= np.sqrt(np.sum(ww**2))
            mm = planar_geometry.rotate_v_into_w_2d(vv, ww, already_normed=True)
            test = np.dot(mm, vv)
            np.testing.assert_allclose(test, ww,
                                       atol=1.0e-10, rtol=1.0e-10)


class TestPlane(unittest.TestCase):

    def test_plane_from_points(self):
        rng = np.random.RandomState(88)
        p1 = rng.random_sample(3)
        p2 = rng.random_sample(3)-0.5
        p3 = rng.random_sample(3)
        p = planar_geometry.Plane.plane_from_points(p1, p2, p3)
        self.assertIsInstance(p, planar_geometry.Plane)
        np.testing.assert_array_equal(p.origin, p3)
        self.assertTrue(p.in_plane(p1))
        self.assertTrue(p.in_plane(p2))
        self.assertTrue(p.in_plane(p3))
        p4 = p1+0.2*p.normal
        self.assertFalse(p.in_plane(p4))

if __name__ == "__main__":
    unittest.main()
