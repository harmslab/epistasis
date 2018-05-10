#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command, Extension
import numpy

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
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "lmfit",
    "notebook",
    "ipython",
    "ipywidgets",
    "scikit-learn",
    "gpmap==0.4.1",
    "emcee"
]

# Extensions
extension = Extension('epistasis.matrix_cython',
                      ['epistasis/matrix_cython.c'],
                      include_dirs=[numpy.get_include()])

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


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


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
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    ext_modules=[extension],
    install_requires=REQUIRED,
    include_package_data=True,
    license='UNLICENSE',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
    keywords='epistasis high-order genetics genotype-phenotype-maps'
)
