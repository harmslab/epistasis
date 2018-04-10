from setuptools import setup, Extension
import numpy

# Define extensions
extension = Extension('epistasis.matrix_cython',
                      ['epistasis/matrix_cython.c'],
                      include_dirs=[numpy.get_include()])

# define all packages for distribution
packages = [
    'epistasis',
    'epistasis.models',
    'epistasis.pyplot',
    'epistasis.sampling',
    'epistasis.simulate',
    'epistasis.models.linear'
]

setup(name='epistasis',
      version='0.6.0',
      description='High-order epistasis models for genotype-phenotype maps',
      author='Zach Sailer',
      author_email='zachsailer@gmail.com',
      url='https://github.com/harmslab/epistasis',
      packages=packages,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      ext_modules=[extension],
      zip_safe=False,
      license='UNLICENSE',
      keywords='epistasis high-order genetics genotype-phenotype-maps')
