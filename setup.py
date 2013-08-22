#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy
from setuptools import setup, Extension

include_dirs = [
    "src",
    numpy.get_include(),
    "/usr/local/include",
    "/usr/local/include/eigen3"
]

library_dirs = [
    "/usr/local/lib",
]

libraries = [
    "amd",
    "camd",
    "colamd",
    "ccolamd",
    "cholmod",
    "suitesparseconfig",
    "metis",
    "blas",
    "lapack",
    "cxsparse",
    "protobuf",
    "gflags",
    "glog",
    "ceres",
]

sources = [
    "src/solver.cc",
    "src/kpsf.c",
]

ext = Extension("kpsf._kpsf",
                sources=sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries)

setup(
    name="kpsf",
    version="0.0.0",
    url="https://github.com/dfm/kpsf",
    license="MIT",
    author="Dan Foreman-Mackey",
    author_email="hi@dfm.io",
    install_requires=["numpy"],
    ext_modules=[ext],
    description="Optimize your PSFs",
    long_description="$$ -- duh.",
    packages=["kpsf"],
    include_package_data=True,
    classifiers=[],
)
