#!/usr/bin/env python

from setuptools import setup

setup(name="SteamRoller",
      version="1.0.21",
      description="An experimental framework for text classification experiments",
      author="Tom Lippincott",
      author_email="tom@cs.jhu.edu",
      url="http://hltcoe.jhu.edu",
      maintainer="Tom Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["steamroller",
                "steamroller.tools",
                "steamroller.tasks",
                "steamroller.models",
                "steamroller.metrics",
                "steamroller.plots",
                "steamroller.scons"],
      package_dir={"steamroller" : "src"},
      scripts=["scripts/steamroller"],
      install_requires=["concrete", "scikit-learn>=0.18.1", "scons>=3.0.0", "plotnine", "flask", "valid", "tqdm", "nltk"],
      package_data={"" : ["data/SConstruct", "data/steamroller_config.json"]},
     )
