------
SteamRoller |Logo|
------

------
Getting started with an example
------

First and foremost, steamroller builds upon the [SCons](https://scons.org/), which has a [very in-depth MAN page](https://scons.org/doc/production/HTML/scons-man.html) and [API](https://scons.org/doc/latest/HTML/scons-api/index.html), but a probably-more-useful [user guide](https://scons.org/doc/production/HTML/scons-user/index.html).

Assuming you're starting in an empty directory named after the experiment (e.g. `~/my_experiment`) and have a recent version of Python 3 on your path::

  $ python -m venv local
  $ source local/bin/activate
  (my_experiment) $ pip install pip -U
  (my_experiment) $ pip install steamroller

Steamroller can create a simple dummy project to help get started::

  (my_experiment) $ steamroller --new_project

As it runs, this will print out information about what it's creating, and why.  The only difference from a basic SCons project, as can be seen in the example SConstruct file, is that `Environment` is imported from steamroller rather than scons.  Other than the "--new project" option, the `steamroller` command behaves like the normal `scons` command, and this will run the experiment one step at a time::

  (my_experiment) $ steamroller -Q
  
However, steamroller makes it easy to instead run on a grid with maximal parallelism, simply by setting a few variables (these will depend on the details of your particular grid, and typically you would want to put these variables in "custom.py" so that you can use the simple `steamroller -Q` command)::

  (my_experiment) $ steamroller -Q STEAMROLLER_ENGINE=slurm CPU_QUEUE=parallel CPU_ACCOUNT=tlippin1 GPU_QUEUE=parallel GPU_ACCOUNT=tlippin1 GPU_COUNT=0

This will submit the experiment to the grid and immediately return: you can check the status of the experiment by running `sacct -s R,PD,F` (work is underway to make monitoring a first-order aspect of steamroller, but for the moment you want to be careful not to resubmit while jobs from a previous invocation are still pending).  See below for the broader set of steamroller variables that can be overridden.

------
Scaling up
------

As described above, steamroller is indistinguishable from the [SCons](https://scons.org/) system itself, which is very well-documented: so, one could simply use it in this fashion, running experiments in serial, locally (e.g. on a laptop).  Steamroller's power comes from the ability to easily flip a switch to take advantage of the massively parallel architecture of a high-performance compute cluster, without changing the underlying code.

When following the pattern described above, this is accomplished by creating a "custom.py" file in the experiment directory and setting a few variables in it.  The most important is `STEAMROLLER_ENGINE`, which defaults to "local", but can alternatively be set to "slurm", "univa", or "sge" (depending on what grid system is used).  There are a few other special variables, all starting with "STEAMROLLER", though bear in mind these can also be set when a particular build rule is called (see, above, how the "PARAM" and "SPLIT" variables are being set)::


  STEAMROLLER_ACCOUNT = "my_account"
  STEAMROLLER_QUEUE = "some_queue"
  STEAMROLLER_TIME = "06:00:00"
  STEAMROLLER_MEMORY = "64G"
  STEAMROLLER_GPU_COUNT = 1

There are a few variables that steamroller uses internally and that you should only set if you *really* know what you're doing::

  STEAMROLLER_SUBMIT_STRING
  STEAMROLLER_NAME_PREFIX
  STEAMROLLER_NAME
  STEAMROLLER_LOG
  STEAMROLLER_WORKING_DIRECTORY
  STEAMROLLER_DEPENDENCIES
  STEAMROLLER_SHELL

Basically, if there's no grid (e.g. you're running on a laptop), steamroller should just behave like SCons and run each needed task as a simple command-line invocation, in linear order that respects the dependencies.  If a grid is specified (e.g. `STEAMROLLER_ENGINE = "slurm"`), and is in fact available, steamroller will instead *submit* each task in the appropriate order, propagating the *task IDs* as appropriate so that the grid jobs respect the dependency structure.

----
FAQ
----

.. |Logo|   image:: logo.png
