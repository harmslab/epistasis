from distutils.core import setup
from Cython.Build import cythonize
import numpy

# To build extension, run python setup.py build_ext --inplace
setup(
    ext_modules=cythonize("matrix_cython.pyx"),
    include_dirs=[numpy.get_include()]
)
