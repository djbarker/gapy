import numpy as np
import unittest

from gapy.ga3d import *

class TestExample(unittest.TestCase):
    """
    Test the example shown in README.md runs
    """

    def test_example(self):

        # construct some basis vectors
        ex = Multivector3D.make.vector(1, 0, 0)
        ey = Multivector3D.make.vector(0, 1, 0)
        ez = Multivector3D.make.vector(0, 0, 1)

        # perform some operations, including geometric product
        Bxy = ex * ey   # a unit bivector
        a = 2*ex + ey   # a vector
        b = 3*ex - ez   # another vector
        M = a*b         # a mixed grade multivector
        M.project(0)    # its scalar projection (inner product)
        M.project(2)    # its bivector project (exterior product)

        # form rotors easily
        Rl = Multivector3D.make.rotor(np.pi/2.0, Bxy)  # a rotor for rotating by 90 degrees in the xy plane
        Rr = Rl.reverse                                # the reverse, needed for right multiplying
        c = Rl*a*Rr                                    # the vector a rotated in the xy plane by 90 degrees
        d = a.rotate_deg(90, Bxy)                      # ... or just use the convenience function

