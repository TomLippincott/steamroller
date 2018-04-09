#!/usr/bin/env python

from setuptools import setup

setup(name="SteamRoller",
      version="1.2.0",
      description="An experimental framework for text classification experiments",
      author="Tom Lippincott",
      author_email="tom@cs.jhu.edu",
      url="http://hltcoe.jhu.edu",
      maintainer="Tom Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["steamroller",
                "steamroller.tools",
                "steamroller.ui",
                #"steamroller.experiments",
                #"steamroller.data_sets",
                "steamroller.feature_extractors",
                "steamroller.feature_transformers",
                "steamroller.model_types",
                "steamroller.measurements",
                "steamroller.visualizations",
                "steamroller.scons"],
      package_dir={"steamroller" : "src"},
      scripts=["scripts/steamroller"],
      install_requires=["scikit-learn>=0.18.1", "scons>=3.0.0", "plotnine", "flask", "drmaa"],
      package_data={"" : ["static_files/SConstruct", "static_files/steamroller_config.json"]},
     )
