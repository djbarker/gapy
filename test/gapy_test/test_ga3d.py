import unittest
import numpy as np

from gapy.ga3d import *

class Test3D(unittest.TestCase):
    """
    Bit of a grab-bag of hand-made tests. 
    Some of them could be systematized.
    """

    def test_equality(self):
        for i in range(0, 8):
            for j in range(0, 8):
                x = Multivector3D(*[1 if ii==i else 0 for ii in range(8)])
                y = Multivector3D(*[1 if jj==j else 0 for jj in range(8)])
                if i == j:
                    self.assertEqual(x, y)
                else:
                    self.assertNotEqual(x, y)

    def test_basic_multiplication(self):
        s = Multivector3D.make.scalar(1)
        x = vec(1, 0, 0)
        y = vec(0, 1, 0)
        B = bivec(1, 0, 0)
        A = bivec(0, 0, 1)
        I = Multivector3D.make.pseudoscalar(1)
        self.assertEqual(s*s,  s)
        self.assertEqual(s*x,  x)
        self.assertEqual(s*y,  y)
        self.assertEqual(s*B,  B)
        self.assertEqual(x*x,  s)
        self.assertEqual(y*y,  s)
        self.assertEqual(I*I, -s)
        self.assertEqual(x*y,  B)
        self.assertEqual(y*x, -B)
        self.assertEqual(x*B,  y)
        self.assertEqual(B*x, -y)
        self.assertEqual(y*B, -x)
        self.assertEqual(B*y,  x)
        self.assertEqual(x*I,  A)
        self.assertEqual(I*x,  A)

    def test_scalar_multiplication(self):
        self.assertEqual( 1*vec(1, 1, 1), vec( 1,  1,  1))
        self.assertEqual( 2*vec(1, 1, 1), vec( 2,  2,  2))
        self.assertEqual(-2*vec(1, 1, 1), vec(-2, -2, -2))

    def test_add_sub(self):
        s = Multivector3D.make.scalar(1)
        x = vec(1, 0, 0)
        y = vec(0, 1, 0)
        I = Multivector3D.make.pseudoscalar(1)
        self.assertEqual(x + x, vec(2, 0, 0))
        self.assertEqual(x + y, vec(1, 1, 0))
        self.assertEqual(s + x, Multivector3D(1, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual(I - x, Multivector3D(0, -1, 0, 0, 0, 0, 0, 1))

    def test_grade(self):
        self.assertEqual(Multivector3D.make.scalar(1).grade, 0)
        self.assertEqual(vec(1, 0, 0).grade, 1)
        self.assertEqual(bivec(1, 0, 0).grade, 2)
        self.assertTrue(np.isnan((vec(1, 0, 0) + bivec(1, 0, 0)).grade))

    def test_projection(self):
        a = Multivector3D(1, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(a.project(0), Multivector3D(1, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(a.project(1), Multivector3D(0, 2, 3, 4, 0, 0, 0, 0))
        self.assertEqual(a.project(2), Multivector3D(0, 0, 0, 0, 5, 6, 7, 0))
        self.assertEqual(a.project(3), Multivector3D(0, 0, 0, 0, 0, 0, 0, 8))

    def test_unit(self):
        a = vec(0.5, 0.5, 0.5)
        A = bivec(0.5, 0.5, 0.5)
        b = a.unit
        B = A.unit
        self.assertAlmostEqual((b*b).scalar, 1.0)
        self.assertAlmostEqual((b*b).scalar, 1.0)

    def test_multiplication(self):
        s = Multivector3D.make.scalar(2)
        a = vec(0.5, 0.5, 0.0)
        b = vec(0, 1, 0)
        I = bivec(1, 0, 0)
        self.assertAlmostEqual((a*b).scalar,          np.cos(np.pi/4) * np.sqrt(0.5))
        self.assertAlmostEqual((a*b).coefficients[4], np.sin(np.pi/4) * np.sqrt(0.5))
        self.assertEqual((a*b).project(1), vec(0, 0, 0))

    def test_reflection(self):
        x = vec(1, 0, 0)
        y = vec(0, 1, 0)
        z = vec(0, 0, 1)
        self.assertEqual(x*y*x, -y)
        self.assertEqual(z*y*z, -y)
        self.assertEqual(y*x*y, -x)
        self.assertEqual(z*x*z, -x)
        self.assertEqual(y*z*y, -z)
        self.assertEqual(x*z*x, -z)

    def test_rotation(self):

        x = vec(2, 0, 0)
        y = vec(0, 3, 0)
        B = y*x  # not unit, rotor should handle that!
        a = vec(0.5, 0.5, 1)

        self.assertEqual(a.rotate_deg( 360,  B), vec(0.5, 0.5, 1.0))
        self.assertEqual(a.rotate_deg( 360, -B), vec(0.5, 0.5, 1.0))
        self.assertEqual(a.rotate_deg( 180,  B), vec(-0.5, -0.5, 1.0))
        self.assertEqual(a.rotate_deg( 180, -B), vec(-0.5, -0.5, 1.0))
        self.assertEqual(a.rotate_deg(-180,  B), vec(-0.5, -0.5, 1.0))
        self.assertEqual(a.rotate_deg( 90,  B), vec(-0.5, 0.5, 1.0))
        self.assertEqual(a.rotate_deg( 90, -B), vec(0.5, -0.5, 1.0))
        self.assertEqual(a.rotate_deg(-90,  B), vec(0.5, -0.5, 1.0))
        self.assertEqual(a.rotate_deg( 45,  B), vec(0, np.sqrt(0.5), 1.0))
        self.assertEqual(a.rotate_deg( 45, -B), vec(np.sqrt(0.5), 0, 1.0))
        self.assertEqual(a.rotate_deg(-45,  B), vec(np.sqrt(0.5), 0, 1.0))


