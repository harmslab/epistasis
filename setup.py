# Try using setuptools first, if it's installed
try:
    from setuptools import setup
except:
    from distutils.core import setup

from distutils.extension import Extension


# Define extensions
extension1 = Extension('epistasis.model_matrix_ext', ["epistasis/model_matrix_ext.c"])

# define all packages for distribution
packages = [
    'epistasis',
    'epistasis.models',
    'epistasis.plot',
    'epistasis.sampling',
    'epistasis.simulate',
]

setup(name='epistasis',
      version='0.2.0',
      description='High Order Epistasis Models/Regressions for Genotype-Phenotype Maps',
      author='Zach Sailer',
      author_email='zachsailer@gmail.com',
      url='https://github.com/harmslab/epistasis',
      packages=packages,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
      ext_modules=[extension1],
      zip_safe=False,
      license='UNLICENSE',
      keywords='epistasis high-order genetics')
