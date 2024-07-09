#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="MRE-based brain age predictions",
    version="0.0.1",
    description="3D Convolutional Neural Network for brain map processing",
    author="Cesar Claros",
    author_email="cesar@udel.edu",
    url="https://github.com/cesar-claros/brain_age",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
