import numpy as np
import matplotlib as plt

from gapy.core import multivector_type

__all__ = ("Multivector3D", "vec", "bivec", 
            "ex", "ey", "ez",
            "Bxy", "Byz", "Bzx",
            "I",
            )


# basis = {1, e1, e2, e3, e1e2, e3e1, e2e3, e1e2e3}
#         [0,  1,  2,  3,    4,    5,    6,      7]
#
# where B3 = e1e2
#       B2 = e3e1
#       B1 = e2e3
#       I = e1e2e3

grade_mask = np.array([0, 1, 1, 1, 2, 2, 2, 3])

# Use this to pick out and multiply b to get a matrix that can then be multipled with a to get the geometric product in 3D
# The entries are the indices in the multiplication table of the respective basis elements on each row.
basis_sign_idx = np.array([
    [ 0,  1,  2,  3, -4, -5, -6, -7],  # scalar
    [ 1,  0, -4,  5,  2, -3, -7, -6],  # e1
    [ 2,  4,  0, -6, -1, -7,  3, -5],  # e2
    [ 3, -5,  6,  0, -7,  1, -2, -4],  # e3
    [ 4,  2, -1,  7,  0,  6, -5,  3],  # B3
    [ 5, -3,  7,  1, -6,  0,  4,  2],  # B2
    [ 6,  7,  3, -2,  5, -4,  0,  1],  # B1
    [ 7,  6,  5,  4,  3,  2,  1,  0]   # I
])

basis_sign = np.sign(basis_sign_idx)
basis_sign = np.where(basis_sign == 0, 1, basis_sign)  # all b_0 terms are additions
basis_idx  = np.abs(basis_sign_idx)

Multivector3D = multivector_type("Multivector3D", basis_idx, basis_sign, grade_mask)

def vec(x: float, y: float, z: float) -> Multivector3D:
    return Multivector3D(0, x, y, z, 0, 0, 0, 0)

def bivec(z: float, y: float, x: float) -> Multivector3D:
    return Multivector3D(0, 0, 0, 0, z, y, x, 0)

# basis vectors & bivectors:
ex = vec(1, 0, 0)
ey = vec(0, 1, 0)
ez = vec(0, 0, 1)
Bxy = bivec(1, 0, 0)
Byz = bivec(0, 1, 0)
Bzx = bivec(0, 0, 1)
I = Multivector3D.make.pseudoscalar(1)