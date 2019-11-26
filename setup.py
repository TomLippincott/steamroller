#!/usr/bin/env python

from setuptools import setup

setup(name="SteamRoller",
      version="1.3.1",
      description="Grid-awareness for SCons build system",
      author="Tom Lippincott",
      author_email="tom@cs.jhu.edu",
      url="http://hltcoe.jhu.edu",
      maintainer="Tom Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["steamroller"],
      package_dir={"steamroller" : "src"},
      install_requires=["scons>=3.0.0"],
     )
