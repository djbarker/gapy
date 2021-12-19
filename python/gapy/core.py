import numpy as np

from typing import Tuple, Type, Dict, TextIO, Callable


__all__ = ("GenericMultivector", "multivector_type",)


class GenericMultivector:
    """
    Base class for all Multivector types. 
    Does not do anything currently, make it ABC and put methods on it to help completion.
    """
    pass


def multivector_type(name: str, basis_idx: np.ndarray, basis_sign: np.ndarray, grade_mask: np.ndarray) -> Type[GenericMultivector]:
    """
    Construct a GenericMultivector type which uses the basis_idx, basis_sign and grade_mask, arrays given.
    """

    # First we perform some sanity checks on the basis:

    # Arrays must be square and equal in shape
    assert grade_mask.ndim == 1
    assert basis_sign.ndim == 2
    assert basis_idx.ndim == 2
    assert basis_idx.shape[0] == basis_idx.shape[1] 
    assert basis_sign.shape[0] == basis_sign.shape[1] 
    assert basis_sign.shape == basis_idx.shape
    assert basis_idx.shape[0] == grade_mask.shape[0]

    # All elements of the basis must appear once only on each row and each column.
    for i in range(2):
        rows = np.sum(basis_idx == i, 0)
        cols = np.sum(basis_idx == i, 1)
        assert np.min(rows) == 1
        assert np.max(rows) == 1
        assert np.min(cols) == 1
        assert np.max(cols) == 1

    # Now we construct the type which will use basis_idx and basis_sign:

    ndims = np.sum(grade_mask == 1)

    def _meta(_: str, supers: Tuple[Type], attrs: Dict[str, Callable]):
        """
        Construct a type with the name coming from `multivector_type`. 
        """
        return type(name, supers, attrs)

    class Multivector(GenericMultivector, metaclass=_meta):
        """
        A class which wraps the numpy operations to give us operator overloading.
        TODO: some sanity checks that we're not mixing, say, Multivector3D with Multivector2D.
        """

        def __init__(self, *coefficients: Tuple[float]):
            coefficients = np.squeeze(np.array(coefficients))
            assert coefficients.shape == grade_mask.shape, f"[{coefficients.shape!r} != {grade_mask.shape!r}]"
            self.coefficients = coefficients

        def __repr__(self) -> str:
            return f"{name}({self.coefficients!r})"

        def __str__(self) -> str:
            return f"[{self.coefficients}, grade={self.grade}]"

        def __eq__(self, other: "Multivector") -> bool:
            if not isinstance(other, Multivector):
                return False
            else:
                return np.allclose(self.coefficients, other.coefficients)

        def __mul__(self, other: "Multivector") -> "Multivector":
            if isinstance(other, Multivector):
                a = self.coefficients
                b = other.coefficients
                B = b[basis_idx] * basis_sign
                return Multivector(B@a)
            else:
                return Multivector(self.coefficients * other)
        
        def __truediv__(self, other: "Multivector") -> "Multivector":
            if isinstance(other, Multivector):
                mag2 = np.sum((other * other).coefficients[0])  # TODO: This is correct for pure vectors & bivectors but what about mixed?
                return self * (other / mag2)
            else:
                return Multivector(self.coefficients / other)

        def __add__(self, other: "Multivector") -> "Multivector":
            return Multivector(self.coefficients + other.coefficients)

        def __sub__(self, other: "Multivector") -> "Multivector":
            return Multivector(self.coefficients - other.coefficients)

        def __neg__(self) -> "Multivector":
            return Multivector(-self.coefficients)

        def __rmul__(self, other: float) -> "Multivector":
            return Multivector.make.scalar(other) * self

        def __radd__(self, other: float) -> "Multivector":
            return Multivector.make.scalar(other) + self

        def __rsub__(self, other: float) -> "Multivector":
            return Multivector.make.scalar(other) - self

        def exp(self) -> "Multivector":
            if self.grade == 0:
                return Multivector.make.scalar(np.exp(self.scalar))
            elif self.grade == 2:
                theta = np.sqrt(np.sum(self.coefficients**2))
                return np.cos(theta) + np.sin(theta)*self.unit
            else:
                raise ValueError(f"Only know how to exponentiate scalars and bivectors! [${B!r}]")

        @property
        def scalar(self) -> float:
            f"""
            Convenience function for projecting onto the zero (scalar) grade.
            :return: Unlike {name}.project, this returns a float, not a {name} object.
            """
            return self.coefficients[0]

        @property
        def vector(self) -> np.ndarray:
            f"""
            Convenience function for projecting onto the vector grade.
            :return: Unlike {name}.project, this returns a raw numpy array, not a {name} object.
            """
            return self.coefficients[grade_mask == 1]

        @property
        def pseudoscalar(self) -> float:
            f"""
            Convenience function for projecting onto the pseudoscalar grade.
            :return: Unlike {name}.project, this returns a float, not a {name} object.
            """
            return self.coefficients[-1]

        def project(self, grade: int) -> "Multivector":
            f"""
            Project onto an arbitrary grade. 
            :return: A {name} object, even if the grade is zero, for example. 
            """
            c_ = [c if grade_mask[i] == grade else 0 for (i, c) in enumerate(self.coefficients)]
            return Multivector(*c_)

        def is_grade(self, i: int) -> bool:
            return np.any(np.abs(self.coefficients[grade_mask == i]) > 1e-8)

        @property
        def grade(self) -> float:
            grades = [self.is_grade(i) for i in range(np.max(grade_mask)+1)]
            if not any(grades):
                return 0  # I _think_ it makes sense that grade(0) = 0
            elif sum(grades) > 1:
                return np.nan  # mixed grade
            else:
                return grades.index(True)

        @property
        def unit(self) -> "Multivector":
            g = self.grade
            if np.isnan(g):
                raise ValueError(f"Cannot unit mixed grade object. [self={self!r}]")
            a = self.coefficients
            return Multivector(a / np.sqrt(np.sum(a*a)))

        @property
        def reverse(self) -> "Multivector":
            return Multivector(*(basis_sign[0, :] * self.coefficients))

        def rotate_rad(self, angle: float, plane: "Multivector") -> "Multivector":
            Rl = Multivector.make.rotor(angle, plane)
            Rr = Rl.reverse
            return Rl*self*Rr

        def rotate_deg(self, angle: float, plane: "Multivector") -> "Multivector":
            return self.rotate_rad(angle * (np.pi/180.), plane)

        class make:
            """
            A helper type to construct Multivectors from a subset of the basis elements.
            """
            
            @staticmethod
            def scalar(a: float) -> "Multivector":
                return Multivector(a, *([0] * (len(grade_mask)-1)))

            @staticmethod
            def vector(*args: float) -> "Multivector":
                assert len(args) == ndims
                return Multivector(0, *args, *[0]*(len(grade_mask) - ndims - 1))

            @staticmethod
            def pseudoscalar(a: float) -> "Multivector":
                return Multivector(*([0] * (len(grade_mask)-1)), a)

            @staticmethod
            def rotor(angle_radians: float, plane: "Multivector") -> "Multivector":
                """
                Returns the (left) Rotor for rotation by the specified angle in the specified plane.
                """
                assert plane.grade == 2, f"Expected bivector! [plane={plane!r}, grade={plane.grade!r}]"
                angle_radians /= 2
                Rl = Multivector.make.scalar(np.cos(angle_radians)) - np.sin(angle_radians) * plane.unit
                return Rl


    return Multivector
