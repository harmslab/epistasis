from setuptools import setup, Extension

# Define extensions
extension = Extension('epistasis.model_matrix_ext',
                      ['epistasis/model_matrix_ext.c'])

# define all packages for distribution
packages = [
    'epistasis',
    'epistasis.models',
    'epistasis.pyplot',
    'epistasis.sampling',
    'epistasis.simulate',
]

setup(name='epistasis',
      version='0.5.0',
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
