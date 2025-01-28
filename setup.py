#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='field-map-ai',
    version='0.0.1',
    description='Train machine learning models to undistort fMRIs',
    author='Jan Tagscherer',
    author_email='jan.tagscherer@nru.dk',
    url='https://github.com/jtagscherer/field-map-ai',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
