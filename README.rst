------
SteamRoller |Logo|
------

SteamRoller is a framework for testing the performance of various machine learning models on different tasks in the broad area of "text classification".  It is designed to make it extremely easy to define new classification tasks, and new models, and drop them in to compare ther characteristics.  It discourages doing anything "special" for different tasks, models, or combinations thereof, to ensure the comparisons are fair and expose all the costs incurred by the different choices.

SteamRoller's user-facing functionality is reflected by four submodules:

1. tasks
2. models
3. metrics
4. plots

Under the hood, SteamRoller has three additional submodules:

1. tools
2. scons
3. ui

As explained below, changes to the code are usually unnecessary, as the most common conceptual classes (tasks and models) can be extended simply by editing the configuration file.

------
Getting started
------

SteamRoller and its dependencies can be installed with ``pip install steamroller --user``.  An empty directory can be initialized for performing experiments by executing ``steamroller init`` from therein.  This creates two files: *SConstruct*, and *steamroller_config.py*.  You can then run ``steamroller run`` to perform the predefined experiments, and ``steamroller serve`` to launch the results web server.  Most of SteamRoller's extensibility is through editing *steamroller_config.py*, while more advanced users may find it useful to edit *SConstruct*.

----
Using an HPC Grid
----

By default, *steamroller_config.py* will set ``GRID=False``, and experiments will run serially on the local machine.  If you are running on an HPC grid like Univa, Sun Grid Engine, or Torque, setting ``GRID=True`` instructs SteamRoller to run experiments via the *qsub* command.  Since the jobs are distributed across the grid, the invocation of SteamRoller will submit them and then *wait* until they have completed, polling the scheduler and printing the current number of running jobs.  If you interrupt the SteamRoller command in this state, *the grid jobs will continue to run*, so you can either allow them to do so (e.g. if the interruption was accidental), or manually kill the running jobs with a command like ``qdel -u USERNAME``.  The latter is particularly important if you want to change and rerun experiments, as otherwise you may have multiple jobs simultaneously building the same output file.

----
Viewing results
----

Once the experiments have finished, you will want to compare their performance.  

----
Defining a new task
----

----
Defining a new model
----

----
FAQ
----

.. |Logo|   image:: logo.png
