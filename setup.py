#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command, Extension

# Package meta-data.
NAME = 'epistasis'
DESCRIPTION = 'A Python API for estimating statistical high-order epistasis in genotype-phenotype maps.'
URL = 'https://github.com/harmslab/epistasis'
EMAIL = 'zachsailer@gmail.com'
AUTHOR = 'Zachary R. Sailer'
REQUIRES_PYTHON = '>=3.3.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    "cython",
    "numpy>=1.15.2",
    "pandas>=0.24.2",
    "scikit-learn>=0.20.0",
    "scipy>=1.1.0",
    "emcee>=2.2.1",
    "lmfit>=0.9.11",
    "matplotlib>=3.0.0",
    "gpmap>=0.6.0",
]

# Hanlding a Cython extension is a pain! Need to
# import numpy before it's installed... Used this
# Stackoverflow solution:
# https://stackoverflow.com/a/42163080

try:
    from Cython.setuptools import build_ext
except:
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    from setuptools.command.build_ext import build_ext
    extension = Extension("epistasis.matrix_cython", ["epistasis/matrix_cython.c"])
else:
    # If we successfully imported Cython, look for a .pyx file
    extension = Extension("epistasis.matrix_cython", ["epistasis/matrix_cython.pyx"])

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    ext_modules=[extension],
    install_requires=REQUIRED,
    extras_require = {
        'test': ['pytest'],
    },
    include_package_data=True,
    license='UNLICENSE',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],
    keywords='epistasis high-order genetics genotype-phenotype-maps',
    cmdclass = {'build_ext': CustomBuildExtCommand},
)
