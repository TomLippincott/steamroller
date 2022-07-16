#!/usr/bin/env python

from setuptools import setup

setup(
    name="SteamRoller",
    version="2.0.0",
    description="Grid-awareness for SCons build system",
    author="Tom Lippincott",
    author_email="tom.lippincott@jhu.edu",
    url="https://github.com/TomLippincott/steamroller",
    maintainer="Tom Lippincott",
    maintainer_email="tom.lippincott@jhu.edu",
    packages=["steamroller"],
    install_requires=["scons>=3.0.0"],
)
