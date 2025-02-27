# -*- coding: utf-8 -*- #
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np,os


from distutils.sysconfig import get_python_inc
np_inc = get_python_inc(plat_specific=True)
# define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

print("Remove build directory and utils_plus.c(python) file")
os.system('rm -rf build')
os.system('rm -rf utils_plus.c*')

your_path = os.path.abspath('./csrc/local')

ext = Extension(name="utils_plus",
                sources=["utils_plus.pyx"],
                libraries=["metis","GKlib"],
                #extra_link_args=['-Wl,-rpath=c/lib']

                include_dirs=[f'{your_path}/include',np.get_include()],
                library_dirs=[f'{your_path}/lib',np_inc],
                )

setup(ext_modules=cythonize(ext, language_level=3))

# python setup.py build_ext --inplace