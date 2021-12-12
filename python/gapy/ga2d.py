import numpy as np
import matplotlib as plt

from gapy.core import multivector_type

__all__ = ("Multivector2D", "scalar", "vec", "bivec")


# basis = {1, e1, e2, e1e2}
#         [0,  1,  2,    3]
#
# where I = e1e2

grade_mask = np.array([0, 1, 1, 2])

# Use this to pick out and multiply b to get a matrix that can then be multipled with a to get the geometric product in 3D
# The entries are the indices in the multiplication table of the respective basis elements on each row.
basis_sign_idx = np.array([
    [  0,  1,  2, -3],  # scalar
    [  1,  0, -3,  2],  # e1
    [  2,  3,  0, -1],  # e2
    [  3,  2, -1,  0]   # I
])

basis_sign = np.sign(basis_sign_idx)
basis_sign = np.where(basis_sign == 0, 1, basis_sign)  # all b_0 terms are additions
basis_idx  = np.abs(basis_sign_idx)

Multivector2D = multivector_type("Multivector2D", basis_idx, basis_sign, grade_mask)

def scalar(a: float) -> Multivector2D:
    return Multivector2D(a, 0, 0, 0)

def vec(x: float, y: float) -> Multivector2D:
    return Multivector2D(0, x, y, 0)

def bivec(b: float) -> Multivector2D:
    return Multivector2D(0, 0, 0, b)
