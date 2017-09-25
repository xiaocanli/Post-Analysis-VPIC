from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'multilinear_cython',
    ext_modules = cythonize("multilinear_cython.pyx"),
    )
