from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyrft',
    install_requires=[
        'numpy',
        'sanssouci',
        'matplotlib',
        'warnings',
        'scipy',
    ],
    version = '0.0.1',
    license='MIT',
    author='Nils PEYROUSET',
    download_url='https://github.com/pilsneyrouset/PHB/',
    author_email='nils.peyrouset@ensae.fr',
    url='https://github.com/pilsneyrouset/PHB/',
    long_description=long_description,
    description='Post-hoc inference',
    packages=find_packages(),
    python_requires='>=3',
)
