#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

NAME = 'masterthesis'
DESCRIPTION = 'Semantic segmentation of WorldView 3 satelite images'
URL = 'https://github.com/boren/MasterThesis'
EMAIL = 'fredrik@bore.ai'
AUTHOR = 'Fredrik Bore, Andreas Taraldsen'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

REQUIRED = [
    'Pillow',
    'h5py',
    'keras>=2.1',
    'git+git://github.com/keras-team/keras-contrib.git@4f07dc3d4c98fd0dd90ba93a87d50b920d8d51e7#egg=keras-contrib',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pandas',
    'pydot',
    'requests',
    'seaborn',
    'scikit-image',
    'scikit-learn',
    'shapely',
    'tifffile==0.13.*',
    'tqdm',
    'webcolors'
]

setup(
    name=NAME,
    version='1.0.0',
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
