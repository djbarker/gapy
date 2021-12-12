import unittest
import numpy as np

from gapy.ga2d import *

class Test2D(unittest.TestCase):
    """
    Bit of a grab-bag of hand-made tests. 
    Some of them could be systematized.
    """

    def test_equality(self):
        for i in range(0, 4):
            for j in range(0, 4):
                x = Multivector2D(*[1 if ii==i else 0 for ii in range(4)])
                y = Multivector2D(*[1 if jj==j else 0 for jj in range(4)])
                if i == j:
                    self.assertEqual(x, y)
                else:
                    self.assertNotEqual(x, y)

    def test_basic_multiplication(self):
        s = Multivector2D.make.scalar(1)
        x = vec(1, 0)
        y = vec(0, 1)
        I = bivec(1)
        self.assertEqual(s*s,  s)
        self.assertEqual(s*x,  x)
        self.assertEqual(s*y,  y)
        self.assertEqual(s*I,  I)
        self.assertEqual(x*x,  s)
        self.assertEqual(y*y,  s)
        self.assertEqual(I*I, -s)
        self.assertEqual(x*y,  I)
        self.assertEqual(y*x, -I)
        self.assertEqual(x*I,  y)
        self.assertEqual(I*x, -y)
        self.assertEqual(y*I, -x)
        self.assertEqual(I*y,  x)

    def test_scalar_multiplication(self):
        self.assertEqual( 1*vec(1, 1), vec( 1,  1))
        self.assertEqual( 2*vec(1, 1), vec( 2,  2))
        self.assertEqual(-2*vec(1, 1), vec(-2, -2))

    def test_add_sub(self):
        s = scalar(1)
        x = vec(1, 0)
        y = vec(0, 1)
        I = bivec(1)
        self.assertEqual(x + x, vec(2, 0))
        self.assertEqual(x + y, vec(1, 1))
        self.assertEqual(s + x, Multivector2D(1, 1, 0, 0))
        self.assertEqual(I - x, Multivector2D(0, -1, 0, 1))

    def test_grade(self):
        self.assertEqual(scalar(1).grade, 0)
        self.assertEqual(vec(1, 0).grade, 1)
        self.assertEqual(bivec(1).grade, 2)
        self.assertTrue(np.isnan((vec(1, 0) + bivec(1)).grade))

    def test_projection(self):
        a = Multivector2D(1, 2, 3, 4)
        self.assertEqual(a.project(0), Multivector2D(1, 0, 0, 0))
        self.assertEqual(a.project(1), Multivector2D(0, 2, 3, 0))
        self.assertEqual(a.project(2), Multivector2D(0, 0, 0, 4))

    def test_unit(self):
        a = vec(0.5, 0.5)
        A = bivec(0.5)
        b = a.unit
        B = A.unit
        self.assertAlmostEqual((b*b).scalar, 1.0)
        self.assertAlmostEqual((b*b).scalar, 1.0)

    def test_multiplication(self):
        s = scalar(2)
        a = vec(0.5, 0.5)
        b = vec(0, 1)
        I = bivec(1)
        self.assertAlmostEqual((a*b).scalar,       np.cos(np.pi/4) * np.sqrt(0.5))
        self.assertAlmostEqual((a*b).pseudoscalar, np.sin(np.pi/4) * np.sqrt(0.5))
        self.assertEqual((a*b).project(1), vec(0, 0))

    def test_reflection(self):
        x = vec(1, 0)
        y = vec(0, 1)
        self.assertEqual(x*y*x, -y)
        self.assertEqual(y*x*y, -x)

    def test_rotation(self):

        x = vec(2, 0)
        y = vec(0, 3)
        B = x*y  # not unit, rotor should handle that!
        a = vec(0.5, 0.5)

        self.assertEqual(a.rotate_deg( 360,  B), vec(0.5, 0.5))
        self.assertEqual(a.rotate_deg( 360, -B), vec(0.5, 0.5))
        self.assertEqual(a.rotate_deg( 180,  B), vec(-0.5, -0.5))
        self.assertEqual(a.rotate_deg( 180, -B), vec(-0.5, -0.5))
        self.assertEqual(a.rotate_deg(-180,  B), vec(-0.5, -0.5))
        self.assertEqual(a.rotate_deg( 90,  B), vec(-0.5, 0.5))
        self.assertEqual(a.rotate_deg( 90, -B), vec(0.5, -0.5))
        self.assertEqual(a.rotate_deg(-90,  B), vec(0.5, -0.5))
        self.assertEqual(a.rotate_deg( 45,  B), vec(0, np.sqrt(0.5)))
        self.assertEqual(a.rotate_deg( 45, -B), vec(np.sqrt(0.5), 0))
        self.assertEqual(a.rotate_deg(-45,  B), vec(np.sqrt(0.5), 0))

