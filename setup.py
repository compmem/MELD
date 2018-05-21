
#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the meld package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from setuptools import setup
from codecs import open
from os import path
import argparse
from distutils.extension import Extension
import numpy as np

# If building on OSX, may need to do this first:
#  export CFLAGS="-I/Users/nielsond/miniconda3/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

DEBUG = False
USE_CYTHON = True
if USE_CYTHON:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import get_directive_defaults

if USE_CYTHON and DEBUG:
    print("Using Cython with Debug Flags")
    extensions = [
        Extension('meld.cluster_topdown', ['meld/cluster_topdown.pyx'], define_macros=[('CYTHON_TRACE', '1')]),
        Extension('meld.tfce',
                  ['meld/tfce.pyx'],
                  define_macros=[('CYTHON_TRACE', '1')],
                  include_dirs=[np.get_include()]),
        Extension('meld.pycluster', ['meld/pycluster.pyx'])
    ]
    directive_defaults = get_directive_defaults()
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True
    extensions = cythonize(extensions, gdb_debug=True)

elif USE_CYTHON:
    print("Using Cython")
    extensions = [
        Extension('meld.cluster_topdown', ['meld/cluster_topdown.pyx']),
        Extension('meld.tfce',
                  ['meld/tfce.pyx'],
                  include_dirs=[np.get_include()]),
        Extension('meld.pycluster', ['meld/pycluster.pyx'])
    ]
    extensions = cythonize(extensions)
else:
    extensions = [
        Extension('meld.cluster_topdown', ['meld/cluster_topdown.c']),
        Extension('meld.tfce',
                  ['meld/tfce.cpp'],
                  include_dirs=[np.get_include()]),
        Extension('meld.pycluster', ['meld/pycluster.cpp'])
    ]


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='meld',
    version='0.1.0',

    description='Mixed effects models for large datasets',
    long_description=long_description,

    packages=['meld'],
    package_dir={"meld": "meld"},
    package_data={'meld': ['*.pxd']},

    author=['Per B. Sederberg', 'Dylan M. Nielson'],
    maintainer=['Per B. Sederberg', 'Dylan M. Nielson'],
    maintainer_email=['psederberg@gmail.com', 'dylan.nielson@gmail.com'],
    url=['http://github.com/compmem/meld'],

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
    ],
    keywords='mixed effects models rpy2',
    
    install_requires=['numpy', 'scipy', 'rpy2', 'joblib', 'jinja2', 'pandas'],
    ext_modules=extensions,
    zip_safe=False
    )

