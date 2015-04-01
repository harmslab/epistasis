from setuptools import setup
from distutils.extension import Extension


extension1 = Extension('epistasis.regression_ext', ["epistasis/regression_ext.c"])

setup(name='epistasis',
      version='0.1',
      description='High Order Epistasis Models/Regressions for Genotype-Phenotype Maps',
      author='Zach Sailer',
      author_email='zachsailer@gmail.com',
      packages=['epistasis'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'scikit-learn',
          'networkx'
      ],
      ext_modules=[extension1],
      zip_safe=False)