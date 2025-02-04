#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='fmdc',
    version='0.0.1',
    description='Train machine learning models to undistort fMRIs',
    author='Magnus Lindberg Christensen',
    author_email='magnus.christensen@nru.dk',
    url='https://github.com/Cyrkryta/fmdc-ai-thesis',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
