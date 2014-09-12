#!/usr/bin/env python
# encoding: utf-8

import os
import sys

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

sys.path.insert(0, "builder")
from builder import build_ext
from builder.ceres import Ceres


if __name__ == "__main__":
    import sys
    import numpy
    from Cython.Build import cythonize

    # Publish the library to PyPI.
    if "publish" in sys.argv[-1]:
        os.system("python setup.py sdist upload")
        sys.exit()

    # Choose libraries to link.
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    # Specify the include directories.
    include_dirs = [
        "kpsf",
        numpy.get_include(),
    ]

    # The source files.
    src = [
        "kpsf/kpsf.cc",
        "kpsf/_kpsf.pyx",
    ]

    # Set up the extension.
    extensions = [
        Extension("kpsf._kpsf", sources=src, libraries=[Ceres()],
                  include_dirs=include_dirs),
    ]

    # Hackishly inject a constant into builtins to enable importing of the
    # package before the library is built.
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins
    builtins.__KPSF_SETUP__ = True
    import kpsf

    # Execute the setup command.
    desc = ""
    # desc = open("README.rst").read()
    setup(
        name="kpsf",
        version=kpsf.__version__,
        author="Daniel Foreman-Mackey",
        author_email="danfm@nyu.edu",
        packages=["kpsf"],
        ext_modules=cythonize(extensions),
        # url="http://github.com/dfm/transit",
        # license="MIT",
        # description="A Python library for computing the light curves of "
        #             "transiting planets",
        # long_description=desc,
        # package_data={"": ["README.rst", "LICENSE", "include/*.h", ]},
        # include_package_data=True,
        cmdclass=dict(build_ext=build_ext),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
        ],
    )
