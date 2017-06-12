from distutils.core import setup
from Cython.Build import cythonize


# To build extension, run python setup.py build_ext --inplace
setup(
    ext_modules = cythonize("model_matrix_ext.pyx")
)
