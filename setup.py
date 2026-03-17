from os.path import abspath, join

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

inc_path = np.get_include()
# Add paths for npyrandom and npymath libraries:
lib_path = [
    abspath(join(np.get_include(), '..', '..', 'random', 'lib')),
    abspath(join(np.get_include(), '..', 'lib')),
]

setup(
    ext_modules=cythonize([
        Extension("noiselab.generators._markov", [
            "src/noiselab/generators/_markov.pyx",
        ], include_dirs=[inc_path], library_dirs=lib_path),
        Extension("noiselab.generators._relaxation", [
            "src/noiselab/generators/_relaxation.pyx",
        ], include_dirs=[inc_path], library_dirs=lib_path, libraries=['npymath', 'npyrandom']),
        Extension("noiselab.adev._adev", [
            "src/noiselab/adev/_adev.pyx",
        ], include_dirs=[inc_path], library_dirs=lib_path),
        Extension("noiselab.tdigest._tdigest", [
            "src/noiselab/tdigest/_tdigest.pyx",
            "src/noiselab/tdigest/t-digest-c/src/tdigest.c",
        ], include_dirs=["src/noiselab/tdigest/t-digest-c/src/", inc_path], library_dirs=lib_path),
    ]),
)
