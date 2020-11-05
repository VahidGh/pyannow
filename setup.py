#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='PyANNOW',
    version='0.1.0',
    python_requires='>=3.2',
    packages=find_packages(),
    description='A Python package for simulating Artificial Neural Networks based on the OpenWorm project.',
    author='Vahid Ghayoomie',
    author_email='vahidghayoomi@gmail.com',
    url='https://github.com/VahidGh/pyannow',
    install_requires=['argparse', 'matplotlib', 'numpy', 'scipy', 'requests', 'jupyter', 'itertools'],
)